"""Minimal RunPod REST v1 adapter with a read-only GPU availability gate.

RunPod's REST v1 API is used for inventory, pod creation, and pod deletion. The
documented GraphQL ``gpuTypes/lowestPrice`` query is used only to select capacity
before creation; no GraphQL mutation is implemented here.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
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
POD_VOLUME_GB: Final = 20
ON_DEMAND_PRICE_TOLERANCE_USD: Final = Decimal("0.005")
POD_GPU_ATTESTATION_TIMEOUT_SECONDS: Final = 180.0
POD_GPU_ATTESTATION_POLL_SECONDS: Final = 5.0
POD_CONNECTIVITY_TIMEOUT_SECONDS: Final = 180.0
POD_CONNECTIVITY_FAST_POLL_WINDOW_SECONDS: Final = 30.0
POD_CONNECTIVITY_FAST_POLL_SECONDS: Final = 2.0
POD_CONNECTIVITY_SLOW_POLL_SECONDS: Final = 5.0
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
_ALLOCATION_ERROR_STATUSES: Final = frozenset({404, 409, 422})
_SENSITIVE_JSON_KEY = re.compile(
    r"(?:authorization|cookie|credential|password|secret|token|api[-_]?key)", re.I
)
_SAFE_JSON_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_SAFE_CONTENT_TYPE = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")
_SAFE_ERROR_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_FIELD_PATH = re.compile(r"^[A-Za-z][A-Za-z0-9_.\[\]-]{0,190}$")
_SECRET_REFERENCE = re.compile(r"^\{\{ RUNPOD_SECRET_[a-z0-9_]{1,96} \}\}$")
_CAPACITY_CODE = re.compile(
    r"^(?:NO_CAPACITY|INSUFFICIENT_CAPACITY|NO_AVAILABLE_INSTANCES|"
    r"GPU_UNAVAILABLE|GPU_CAPACITY_UNAVAILABLE|OUT_OF_CAPACITY|"
    r"NO_MACHINE_AVAILABLE)$",
    re.I,
)
_CAPACITY_MESSAGE = re.compile(
    r"(?:\b(?:no|zero)\s+(?:available\s+)?(?:gpu\s+)?(?:instances?|machines?|capacity)\b|"
    r"\binsufficient\s+(?:gpu\s+)?capacity\b|"
    r"\bout\s+of\s+(?:gpu\s+)?capacity\b|"
    r"\bno\s+(?:gpu\s+)?instances?\s+(?:is\s+|are\s+)?available\b)",
    re.I,
)
_VALIDATION_MESSAGE = re.compile(
    r"\b(?:validation|schema|required|missing|invalid|unknown|unexpected|field|"
    r"must\s+be|greater\s+than|less\s+than|not\s+permitted)\b",
    re.I,
)
_CREATE_PAYLOAD_FIELDS: Final = frozenset(
    {
        "name",
        "imageName",
        "cloudType",
        "computeType",
        "gpuTypeIds",
        "gpuTypePriority",
        "gpuCount",
        "interruptible",
        "containerDiskInGb",
        "volumeInGb",
        "volumeMountPath",
        "ports",
        "supportPublicIp",
        "env",
    }
)
_CREATE_ENV_FIELDS: Final = frozenset(
    {
        "MAPILLARY_ACCESS_TOKEN",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "PUBLIC_KEY",
    }
)
_MISSING: Final = object()
_GPU_LIST_QUERY: Final = """query AtlasLensPhase3FGpuTypes {
  gpuTypes {
    id
    displayName
    memoryInGb
  }
}"""
_ACCOUNT_BILLING_QUERY: Final = """query AtlasLensPhase3FAccountBilling {
  myself {
    clientBalance
    currentSpendPerHr
  }
}"""


class RunPodAPIError(RuntimeError):
    """Stable failure code; response bodies, URLs, and credentials are excluded."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class RunPodCreateError(RunPodAPIError):
    """Post-ID create failure carrying the private cleanup target in memory only."""

    def __init__(self, code: str, pod_record: PodRecord) -> None:
        super().__init__(code)
        self.pod_record = pod_record


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


def _json_type(value: object) -> str:
    if value is _MISSING:
        return "missing"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return "array"
    return "unknown"


def _safe_content_type(value: str | None) -> str:
    media_type = "" if value is None else value.split(";", maxsplit=1)[0].strip().lower()
    return media_type if _SAFE_CONTENT_TYPE.fullmatch(media_type) else "unavailable"


def _safe_error_code(value: object) -> str:
    if (
        isinstance(value, str)
        and _SAFE_ERROR_CODE.fullmatch(value)
        and not _SENSITIVE_JSON_KEY.search(value)
    ):
        return value
    return "provider_validation_error"


def _safe_field_path(value: object) -> str:
    if isinstance(value, str):
        if _SAFE_FIELD_PATH.fullmatch(value) and not _SENSITIVE_JSON_KEY.search(value):
            return value
        return "<redacted-field>"
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        parts: list[str] = []
        for item in value:
            if isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= 9999:
                parts.append(f"[{item}]")
            elif (
                isinstance(item, str)
                and _SAFE_JSON_KEY.fullmatch(item)
                and not _SENSITIVE_JSON_KEY.search(item)
            ):
                parts.append(item)
            else:
                return "<redacted-field>"
        rendered = ".".join(parts).replace(".[", "[")
        if rendered and len(rendered) <= 191:
            return rendered
    return "<unavailable-field>"


def _validation_message_class(code: str, message: object) -> str:
    normalized = code.lower()
    text = message.lower() if isinstance(message, str) else ""
    combined = f"{normalized} {text}"
    if "missing" in combined or "required" in combined:
        return "required_field_missing"
    if "extra" in combined or "unknown" in combined or "not permitted" in combined:
        return "unknown_field"
    if "type" in combined or "parsing" in combined:
        return "wrong_json_type"
    if any(term in combined for term in ("greater", "less", "range", "limit")):
        return "value_out_of_range"
    if "enum" in combined or "literal" in combined or "unsupported" in combined:
        return "unsupported_value"
    if "invalid" in combined or "value_error" in combined:
        return "invalid_value"
    return "provider_validation_error"


@dataclass(frozen=True, slots=True)
class RunPodConfig:
    """Immutable pod fields which cannot be supplied by untrusted runtime input."""

    image_name: str
    gpu_type_preferences: tuple[str, ...]
    container_disk_gb: int = MAX_CONTAINER_DISK_GB
    min_gpu_memory_gb: int = MIN_GPU_MEMORY_GB
    ssh_public_key: str | None = None
    allowed_cloud_types: tuple[str, ...] = ("SECURE", "COMMUNITY")

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
    available_gpu_counts: tuple[int, ...] | None

    @property
    def capacity_confirmed(self) -> bool:
        return self.available_gpu_counts is not None and 1 in self.available_gpu_counts

    @property
    def capacity_evidence(self) -> str:
        if self.capacity_confirmed:
            return "available_gpu_counts"
        return "advertised_stock_status"


def validate_create_payload_contract(
    payload: Mapping[str, object],
    *,
    expected_gpu_type_id: str,
    expected_image_name: str,
    expected_cloud_type: str,
) -> CreatePayloadContractReport:
    """Validate the bounded Phase 3F subset of RunPod's official Pod schema."""

    code = "RUNPOD_CREATE_PAYLOAD_INVALID"
    _require(set(payload) == _CREATE_PAYLOAD_FIELDS, code)
    _require(
        isinstance(payload.get("name"), str)
        and bool(_MARKER.fullmatch(cast(str, payload["name"]))),
        code,
    )
    _require(payload.get("imageName") == expected_image_name, code)
    _require(payload.get("cloudType") == expected_cloud_type, code)
    _require(expected_cloud_type in {"SECURE", "COMMUNITY"}, code)
    _require(payload.get("computeType") == "GPU", code)
    gpu_type_ids = payload.get("gpuTypeIds")
    _require(
        isinstance(gpu_type_ids, list)
        and gpu_type_ids == [expected_gpu_type_id]
        and bool(_GPU_TYPE_ID.fullmatch(expected_gpu_type_id)),
        code,
    )
    _require(payload.get("gpuTypePriority") == "custom", code)
    _require(type(payload.get("gpuCount")) is int and payload["gpuCount"] == 1, code)
    _require(payload.get("interruptible") is False, code)
    container_disk = payload.get("containerDiskInGb")
    _require(
        type(container_disk) is int
        and 1 <= container_disk <= MAX_CONTAINER_DISK_GB,
        code,
    )
    volume = payload.get("volumeInGb")
    _require(type(volume) is int and volume == POD_VOLUME_GB, code)
    _require(payload.get("volumeMountPath") == "/workspace", code)
    _require(payload.get("ports") == ["22/tcp"], code)
    _require(payload.get("supportPublicIp") is True, code)
    _require("networkVolumeId" not in payload, code)
    environment = _object(payload.get("env"), code)
    _require(
        set(environment).issubset(_CREATE_ENV_FIELDS)
        and {
            "MAPILLARY_ACCESS_TOKEN",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
        }.issubset(environment),
        code,
    )
    _require(environment.get("MAPILLARY_ACCESS_TOKEN") == MAPILLARY_ENV_REFERENCE, code)
    _require(
        isinstance(environment.get("MAPILLARY_ACCESS_TOKEN"), str)
        and bool(
            _SECRET_REFERENCE.fullmatch(
                cast(str, environment["MAPILLARY_ACCESS_TOKEN"])
            )
        ),
        code,
    )
    _require(environment.get("HF_HUB_OFFLINE") == "1", code)
    _require(environment.get("TRANSFORMERS_OFFLINE") == "1", code)
    public_key = environment.get("PUBLIC_KEY", _MISSING)
    _require(
        public_key is _MISSING
        or (
            isinstance(public_key, str)
            and bool(_SSH_PUBLIC_KEY.fullmatch(public_key))
        ),
        code,
    )
    _require(all(isinstance(value, str) for value in environment.values()), code)
    fields = tuple(sorted((name, _json_type(value)) for name, value in payload.items()))
    env_fields = tuple(
        sorted((str(name), _json_type(value)) for name, value in environment.items())
    )
    return CreatePayloadContractReport(fields=fields, env_fields=env_fields)


def build_create_payload(
    config: RunPodConfig,
    request: PodRequest,
    offer: GPUOffer,
) -> tuple[dict[str, object], CreatePayloadContractReport]:
    environment: dict[str, object] = {
        "MAPILLARY_ACCESS_TOKEN": MAPILLARY_ENV_REFERENCE,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    if config.ssh_public_key is not None:
        environment["PUBLIC_KEY"] = config.ssh_public_key
    payload: dict[str, object] = {
        "name": request.run_marker,
        "imageName": config.image_name,
        "cloudType": offer.cloud_type,
        "computeType": "GPU",
        "gpuTypeIds": [offer.gpu_type_id],
        "gpuTypePriority": "custom",
        "gpuCount": 1,
        "interruptible": False,
        "containerDiskInGb": config.container_disk_gb,
        "volumeInGb": POD_VOLUME_GB,
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        "supportPublicIp": True,
        "env": environment,
    }
    report = validate_create_payload_contract(
        payload,
        expected_gpu_type_id=offer.gpu_type_id,
        expected_image_name=config.image_name,
        expected_cloud_type=offer.cloud_type,
    )
    return payload, report


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
                    "availableGpuCounts": (
                        None
                        if selected.available_gpu_counts is None
                        else list(selected.available_gpu_counts)
                    ),
                    "capacity_confirmed": selected.capacity_confirmed,
                    "capacity_evidence": selected.capacity_evidence,
                }
            ),
            "selected_gpu_id": None if selected is None else selected.gpu_type_id,
            "selected_gpu_display_name": (
                None if selected is None else selected.display_name
            ),
            "selected_hourly_price": (
                None if selected is None else str(selected.hourly_price)
            ),
            "selected_stock_status": (
                None if selected is None else selected.stock_status
            ),
            "capacity_confirmed": (
                None if selected is None else selected.capacity_confirmed
            ),
            "capacity_evidence": (
                None if selected is None else selected.capacity_evidence
            ),
            "create_attempt_limit": 1,
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
class CreateValidationErrorDiagnostic:
    field_path: str
    type_code: str
    message_class: str

    def to_public_dict(self) -> dict[str, str]:
        return {
            "field_path": self.field_path,
            "type_code": self.type_code,
            "message_class": self.message_class,
        }


@dataclass(frozen=True, slots=True)
class CreateResponseDiagnostic:
    http_status: int
    top_level_keys: tuple[str, ...]
    id_present: bool | None
    interruptible_present: bool | None
    interruptible_json_type: str
    classification: str
    body_kind: str
    byte_length: int
    sha256: str | None
    content_type: str
    failure_code: str | None
    validation_errors: tuple[CreateValidationErrorDiagnostic, ...] = ()
    secret_free: bool = True

    def to_public_dict(self) -> dict[str, object]:
        return {
            "http_status": self.http_status,
            "response_top_level_keys": list(self.top_level_keys),
            "id_present": self.id_present,
            "interruptible_present": self.interruptible_present,
            "interruptible_json_type": self.interruptible_json_type,
            "response_classification": self.classification,
            "response_body_kind": self.body_kind,
            "response_byte_length": self.byte_length,
            "response_sha256": self.sha256,
            "response_content_type": self.content_type,
            "failure_code": self.failure_code,
            "validation_errors": [
                item.to_public_dict() for item in self.validation_errors
            ],
            "secret_free": self.secret_free,
        }


@dataclass(frozen=True, slots=True)
class CreatePayloadContractReport:
    fields: tuple[tuple[str, str], ...]
    env_fields: tuple[tuple[str, str], ...]
    valid: bool = True

    def to_public_dict(self) -> dict[str, object]:
        return {
            "payload_fields": [
                {"name": name, "json_type": json_type}
                for name, json_type in self.fields
            ],
            "payload_env_fields": [
                {"name": name, "json_type": json_type}
                for name, json_type in self.env_fields
            ],
            "payload_contract_valid": self.valid,
            "secret_values_included": False,
        }


@dataclass(frozen=True, slots=True)
class PodRentalAttestationDiagnostic:
    evidence: str
    request_interruptible: bool
    create_http_status: int
    selected_gpu_id: str
    selected_uninterruptable_price: Decimal
    create_cost_per_hr: Decimal
    price_delta_usd: Decimal
    desired_status: str
    cloud_type: str
    create_interruptible_present: bool
    create_interruptible_json_type: str
    get_verification_http_status: int | None
    get_interruptible_present: bool | None
    get_interruptible_json_type: str | None
    pod_inventory_count: int | None
    explicit_false_source: str | None
    create_http_class: str
    normalized_gpu_path: str
    gpu_poll_count: int
    gpu_poll_elapsed_seconds: float
    observed_gpu_id: str
    gpu_count: int
    cost_attestation: str
    secret_free: bool = True

    def to_public_dict(self) -> dict[str, object]:
        return {
            "rental_evidence": self.evidence,
            "request_interruptible": self.request_interruptible,
            "create_http_status": self.create_http_status,
            "selected_gpu_id": self.selected_gpu_id,
            "selected_uninterruptable_price": str(
                self.selected_uninterruptable_price
            ),
            "create_cost_per_hr": str(self.create_cost_per_hr),
            "price_delta_usd": str(self.price_delta_usd),
            "desired_status": self.desired_status,
            "cloud_type": self.cloud_type,
            "create_interruptible_present": self.create_interruptible_present,
            "create_interruptible_json_type": self.create_interruptible_json_type,
            "get_verification_http_status": self.get_verification_http_status,
            "get_interruptible_present": self.get_interruptible_present,
            "get_interruptible_json_type": self.get_interruptible_json_type,
            "pod_inventory_count": self.pod_inventory_count,
            "explicit_false_source": self.explicit_false_source,
            "create_http_class": self.create_http_class,
            "normalized_gpu_path": self.normalized_gpu_path,
            "gpu_poll_count": self.gpu_poll_count,
            "gpu_poll_elapsed_seconds": self.gpu_poll_elapsed_seconds,
            "observed_gpu_id": self.observed_gpu_id,
            "gpu_count": self.gpu_count,
            "cost_attestation": self.cost_attestation,
            "secret_free": self.secret_free,
        }


@dataclass(frozen=True, slots=True)
class PodGPUAttestationProgressDiagnostic:
    outcome: str
    failure_code: str | None
    normalized_gpu_path: str | None
    poll_count: int
    poll_elapsed_seconds: float
    final_desired_status: str | None
    expected_gpu_id: str
    observed_gpu_id: str | None
    gpu_count: int | None
    cost_attestation: str | None
    receipt_bound_pod_count: int | None = None
    unexpected_pod_count: int | None = None
    endpoint_count: int | None = None
    network_volume_count: int | None = None
    template_count: int | None = None
    receipt_bound_match: bool | None = None
    observed_gpu_id_sha256: str | None = None
    create_http_class: str = "success_201"
    secret_free: bool = True

    def to_public_dict(self) -> dict[str, object]:
        return {
            "gpu_attestation_outcome": self.outcome,
            "gpu_attestation_failure_code": self.failure_code,
            "normalized_gpu_path": self.normalized_gpu_path,
            "gpu_poll_count": self.poll_count,
            "gpu_poll_elapsed_seconds": self.poll_elapsed_seconds,
            "final_desired_status": self.final_desired_status,
            "expected_gpu_id": self.expected_gpu_id,
            "observed_gpu_id": self.observed_gpu_id,
            "observed_gpu_id_sha256": self.observed_gpu_id_sha256,
            "gpu_count": self.gpu_count,
            "cost_attestation": self.cost_attestation,
            "receipt_bound_pod_count": self.receipt_bound_pod_count,
            "unexpected_pod_count": self.unexpected_pod_count,
            "endpoint_count": self.endpoint_count,
            "network_volume_count": self.network_volume_count,
            "template_count": self.template_count,
            "receipt_bound_match": self.receipt_bound_match,
            "create_http_class": self.create_http_class,
            "secret_free": self.secret_free,
        }


@dataclass(frozen=True, slots=True)
class PodConnectivityProgressDiagnostic:
    outcome: str
    failure_code: str | None
    public_ip_present: bool
    tcp_port_present: bool
    poll_count: int
    elapsed_seconds: float
    ssh_ready: bool
    secret_free: bool = True

    def to_public_dict(self) -> dict[str, object]:
        return {
            "connectivity_outcome": self.outcome,
            "connectivity_failure_code": self.failure_code,
            "public_ip_present": self.public_ip_present,
            "tcp_port_present": self.tcp_port_present,
            "connectivity_poll_count": self.poll_count,
            "connectivity_elapsed_seconds": self.elapsed_seconds,
            "ssh_ready": self.ssh_ready,
            "secret_free": self.secret_free,
        }


@dataclass(frozen=True, slots=True)
class RunPodInventory:
    pods: tuple[PodRecord, ...]
    endpoint_ids: tuple[str, ...]
    network_volume_ids: tuple[str, ...]
    template_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RunPodBillingSnapshot:
    """Sanitized authenticated account totals; the provider body is never retained."""

    client_balance_usd: Decimal
    current_spend_per_hour_usd: Decimal

    def __post_init__(self) -> None:
        _require(
            self.client_balance_usd.is_finite() and self.client_balance_usd >= 0,
            "billing_balance_invalid",
        )
        _require(
            self.current_spend_per_hour_usd.is_finite()
            and self.current_spend_per_hour_usd >= 0,
            "billing_current_spend_invalid",
        )


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
        bind_created_pod: Callable[[PodRecord], None] | None = None,
        record_rental_attestation: (
            Callable[[PodRentalAttestationDiagnostic], None] | None
        ) = None,
        record_gpu_attestation_progress: (
            Callable[[PodGPUAttestationProgressDiagnostic], None] | None
        ) = None,
        record_connectivity_progress: (
            Callable[[PodConnectivityProgressDiagnostic], None] | None
        ) = None,
        require_receipt_binding: bool = False,
        attestation_timeout_seconds: float = POD_GPU_ATTESTATION_TIMEOUT_SECONDS,
        attestation_poll_seconds: float = POD_GPU_ATTESTATION_POLL_SECONDS,
        connectivity_timeout_seconds: float = POD_CONNECTIVITY_TIMEOUT_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        _require(
            bool(api_token)
            and api_token.strip() == api_token
            and "\r" not in api_token
            and "\n" not in api_token,
            "api_token_invalid",
        )
        _require(0 < timeout_seconds <= 30, "timeout_invalid")
        _require(
            0 < attestation_timeout_seconds <= POD_GPU_ATTESTATION_TIMEOUT_SECONDS,
            "attestation_timeout_invalid",
        )
        _require(
            0 < attestation_poll_seconds <= 30,
            "attestation_poll_invalid",
        )
        _require(
            0 < connectivity_timeout_seconds <= POD_CONNECTIVITY_TIMEOUT_SECONDS,
            "connectivity_timeout_invalid",
        )
        _require(isinstance(require_receipt_binding, bool), "receipt_binding_mode_invalid")
        self._api_token = api_token
        self._config = config
        self._last_offer: GPUOffer | None = None
        self._last_availability_report: GPUAvailabilityReport | None = None
        self._last_graphql_errors: tuple[GraphQLErrorDiagnostic, ...] = ()
        self._last_created_hourly_price: Decimal | None = None
        self._last_create_response_diagnostic: CreateResponseDiagnostic | None = None
        self._last_rental_attestation: PodRentalAttestationDiagnostic | None = None
        self._last_gpu_attestation_progress: (
            PodGPUAttestationProgressDiagnostic | None
        ) = None
        self._last_connectivity_progress: PodConnectivityProgressDiagnostic | None = None
        self._bind_created_pod = bind_created_pod
        self._record_rental_attestation = record_rental_attestation
        self._record_gpu_attestation_progress = record_gpu_attestation_progress
        self._record_connectivity_progress = record_connectivity_progress
        self._require_receipt_binding = require_receipt_binding
        self._receipt_binding_confirmed = False
        self._attestation_timeout_seconds = attestation_timeout_seconds
        self._attestation_poll_seconds = attestation_poll_seconds
        self._connectivity_timeout_seconds = connectivity_timeout_seconds
        self._monotonic = monotonic
        self._sleep = sleep
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
    def last_create_response_diagnostic(self) -> CreateResponseDiagnostic | None:
        return self._last_create_response_diagnostic

    @property
    def last_rental_attestation(self) -> PodRentalAttestationDiagnostic | None:
        return self._last_rental_attestation

    @property
    def last_gpu_attestation_progress(
        self,
    ) -> PodGPUAttestationProgressDiagnostic | None:
        return self._last_gpu_attestation_progress

    @property
    def last_connectivity_progress(self) -> PodConnectivityProgressDiagnostic | None:
        return self._last_connectivity_progress

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
        return self._pod_records(rows, duplicate_code="pod_record_invalid")

    @staticmethod
    def _pod_records(
        rows: Sequence[Mapping[str, object]],
        *,
        duplicate_code: str,
    ) -> tuple[PodRecord, ...]:
        by_id: dict[str, PodRecord] = {}
        for row in rows:
            pod_id = _resource_id(row.get("id"), "pod_id_invalid")
            name = row.get("name")
            marker = name if isinstance(name, str) and _MARKER.fullmatch(name) else None
            try:
                candidate = PodRecord(pod_id, marker)
            except Phase3FSafetyError as exc:
                raise RunPodAPIError("pod_record_invalid") from exc
            prior = by_id.get(pod_id)
            if prior is not None:
                _require(
                    prior.run_marker is None
                    or candidate.run_marker is None
                    or hmac.compare_digest(prior.run_marker, candidate.run_marker),
                    duplicate_code,
                )
                if prior.run_marker is None and candidate.run_marker is not None:
                    by_id[pod_id] = candidate
            else:
                by_id[pod_id] = candidate
        return tuple(sorted(by_id.values(), key=lambda item: item.pod_id))

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

    def account_billing_snapshot(self) -> RunPodBillingSnapshot:
        """Read authenticated account billing totals without a cloud mutation."""

        payload = self._request_json(
            "POST",
            GRAPHQL_URL,
            expected_status=200,
            json_body={
                "query": _ACCOUNT_BILLING_QUERY,
                "operationName": "AtlasLensPhase3FAccountBilling",
            },
        )
        root = _object(payload, "billing_query_response_invalid")
        errors = self._sanitize_graphql_errors(root.get("errors"))
        self._last_graphql_errors = errors
        _require(not errors, "BILLING_RECONCILIATION_GRAPHQL_ERRORS")
        data = _object(root.get("data"), "billing_query_response_invalid")
        myself = _object(data.get("myself"), "billing_query_response_invalid")
        _require(
            set(myself) == {"clientBalance", "currentSpendPerHr"},
            "billing_query_response_invalid",
        )
        return RunPodBillingSnapshot(
            client_balance_usd=_decimal(
                myself.get("clientBalance"),
                "billing_balance_invalid",
            ),
            current_spend_per_hour_usd=_decimal(
                myself.get("currentSpendPerHr"),
                "billing_current_spend_invalid",
            ),
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
                self._preference_rank(offer.gpu_type_id),
                offer.hourly_price,
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
                0 if item[0].cloud_type == "SECURE" else 1,
                item[0].hourly_price,
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
        if counts_value is None:
            offer = GPUOffer(
                gpu_type_id=gpu_type_id,
                display_name=display_name,
                memory_gb=memory_gb,
                hourly_price=price,
                stock_status=status,
                cloud_type=cloud_type,
                secure_cloud=secure_cloud,
                community_cloud=community_cloud,
                available_gpu_counts=None,
            )
            diagnostic = GPUCandidateDiagnostic(
                gpu_type_id=gpu_type_id,
                display_name=display_name,
                memory_gb=memory_gb,
                secure_cloud=secure_cloud,
                community_cloud=community_cloud,
                stock_status=status,
                uninterruptable_price=price,
                available_gpu_counts=None,
                accepted=True,
                classification="capacity_unconfirmed_but_advertised",
            )
            return offer, diagnostic
        if not isinstance(counts_value, Sequence) or isinstance(counts_value, str | bytes):
            return reject(
                "available_gpu_counts_invalid",
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
        _require(request.public_ports == (22,), "ssh_port_required")
        _require(self._config.ssh_public_key is not None, "ssh_configuration_mismatch")
        _require(request.gpu_type_id is not None, "gpu_type_id_required")
        offer = self.revalidate_gpu_offer(
            cast(str, request.gpu_type_id),
            max_hourly_price=request.hourly_cost_usd,
        )
        self._last_offer = offer
        self._last_gpu_attestation_progress = None
        self._receipt_binding_confirmed = False
        create_payload, _contract = build_create_payload(self._config, request, offer)
        self._cloud_mutation_count += 1
        payload = self._request_json(
            "POST",
            "pods",
            expected_status=201,
            response_observer=self._observe_create_response,
            status_classifier=self._classify_create_status,
            json_body=create_payload,
        )
        row = _object(payload, "create_response_invalid")
        pod_id = _resource_id(row.get("id"), "create_response_invalid")
        try:
            created = PodRecord(pod_id, request.run_marker)
        except Phase3FSafetyError as exc:
            raise RunPodAPIError("create_response_invalid") from exc
        try:
            if self._bind_created_pod is not None:
                try:
                    self._bind_created_pod(created)
                except Exception:
                    raise RunPodAPIError("POD_RECEIPT_BINDING_MISSING") from None
                self._receipt_binding_confirmed = True
            _require(
                self._receipt_binding_confirmed or not self._require_receipt_binding,
                "POD_RECEIPT_BINDING_MISSING",
            )
            _require(row.get("name") == request.run_marker, "create_response_marker_mismatch")
            _require(row.get("volumeInGb") == POD_VOLUME_GB, "create_response_volume")
            _require(
                row.get("volumeMountPath") == "/workspace",
                "create_response_volume_mount",
            )
            disk = row.get("containerDiskInGb")
            _require(
                isinstance(disk, int) and 1 <= disk <= MAX_CONTAINER_DISK_GB,
                "create_response_disk_invalid",
            )
            returned_ports = _sequence(row.get("ports"), "create_response_ports_invalid")
            _require(set(returned_ports).issubset({"22/tcp"}), "create_response_ports_invalid")
            attestation = self._poll_allocated_pod(
                create_row=row,
                created=created,
                request_interruptible=create_payload.get("interruptible", _MISSING),
                request=request,
                offer=offer,
            )
            self._last_rental_attestation = attestation
            if self._record_rental_attestation is not None:
                self._record_rental_attestation(attestation)
            progress = self._last_gpu_attestation_progress
            if progress is not None:
                self._record_gpu_progress(replace(progress, outcome="attested"))
            self._last_created_hourly_price = attestation.create_cost_per_hr
        except RunPodAPIError as exc:
            self._fail_gpu_attestation_progress(exc.code)
            raise RunPodCreateError(exc.code, created) from None
        except BaseException:
            self._fail_gpu_attestation_progress("create_response_validation_failed")
            raise RunPodCreateError("create_response_validation_failed", created) from None
        return created

    def _poll_allocated_pod(
        self,
        *,
        create_row: Mapping[str, object],
        created: PodRecord,
        request_interruptible: object,
        request: PodRequest,
        offer: GPUOffer,
    ) -> PodRentalAttestationDiagnostic:
        _require(request_interruptible is False, "create_request_interruptible_invalid")
        create_value = create_row.get("interruptible", _MISSING)
        create_present = "interruptible" in create_row
        create_type = _json_type(create_value)
        self._record_gpu_progress(
            PodGPUAttestationProgressDiagnostic(
                outcome="pending",
                failure_code=None,
                normalized_gpu_path=None,
                poll_count=0,
                poll_elapsed_seconds=0.0,
                final_desired_status=None,
                expected_gpu_id=offer.gpu_type_id,
                observed_gpu_id=None,
                gpu_count=None,
                cost_attestation=None,
            )
        )
        _require(
            create_type in {"boolean", "missing", "null", "string"},
            "create_response_interruptible_invalid",
        )
        create_gpu_id, create_gpu_path, create_gpu_count = self._normalized_gpu(
            create_row
        )
        self._record_gpu_progress(
            PodGPUAttestationProgressDiagnostic(
                outcome="pending",
                failure_code=None,
                normalized_gpu_path=create_gpu_path,
                poll_count=0,
                poll_elapsed_seconds=0.0,
                final_desired_status=(
                    cast(str, create_row.get("desiredStatus"))
                    if create_row.get("desiredStatus")
                    in {"RUNNING", "EXITED", "TERMINATED"}
                    else None
                ),
                expected_gpu_id=offer.gpu_type_id,
                observed_gpu_id=(
                    create_gpu_id if create_gpu_id == offer.gpu_type_id else None
                ),
                gpu_count=create_gpu_count,
                cost_attestation=None,
                observed_gpu_id_sha256=(
                    hashlib.sha256(create_gpu_id.encode("utf-8")).hexdigest()
                    if create_gpu_id is not None
                    and create_gpu_id != offer.gpu_type_id
                    else None
                ),
            )
        )
        if create_gpu_id is not None:
            _require(
                create_gpu_id == offer.gpu_type_id,
                "create_response_gpu_mismatch",
            )
        if create_gpu_count is not None:
            _require(create_gpu_count == 1, "create_response_gpu_count_mismatch")
        create_desired = create_row.get("desiredStatus", _MISSING)
        if create_desired is not _MISSING and create_desired is not None:
            _require(
                create_desired == "RUNNING",
                "create_response_desired_status_invalid",
            )
        if create_value is True:
            raise RunPodAPIError("create_response_interruptible_true")
        create_price = self._optional_attested_price(
            create_row,
            offer=offer,
            maximum=request.hourly_cost_usd,
        )
        started = self._monotonic()
        deadline = started + self._attestation_timeout_seconds
        poll_count = 0
        while True:
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                current = self._last_gpu_attestation_progress
                if current is not None:
                    self._last_gpu_attestation_progress = replace(
                        current,
                        poll_elapsed_seconds=self._attestation_timeout_seconds,
                    )
                raise RunPodAPIError("POD_GPU_ATTESTATION_TIMEOUT")
            poll_count += 1
            try:
                payload = self._request_json(
                    "GET",
                    f"pods/{created.pod_id}",
                    expected_status=200,
                    params={"includeMachine": "true", "includeNetworkVolume": "true"},
                    timeout_seconds=min(remaining, 30.0),
                )
            except RunPodAPIError as exc:
                if exc.code == "unexpected_status_404":
                    raise RunPodAPIError("POD_GPU_ATTESTATION_POD_MISSING") from None
                raise
            verified = _object(payload, "pod_gpu_attestation_response_invalid")
            verified_id = _resource_id(
                verified.get("id"),
                "pod_gpu_attestation_response_invalid",
            )
            _require(
                hmac.compare_digest(verified_id, created.pod_id),
                "pod_gpu_attestation_id_mismatch",
            )

            marker = verified.get("name", _MISSING)
            get_value = verified.get("interruptible", _MISSING)
            get_present = "interruptible" in verified
            get_type = _json_type(get_value)
            desired = verified.get("desiredStatus", _MISSING)
            gpu_id, gpu_path, gpu_count = self._normalized_gpu(verified)
            elapsed = max(0.0, self._monotonic() - started)
            self._record_gpu_progress(
                PodGPUAttestationProgressDiagnostic(
                    outcome="pending",
                    failure_code=None,
                    normalized_gpu_path=gpu_path,
                    poll_count=poll_count,
                    poll_elapsed_seconds=min(
                        elapsed, POD_GPU_ATTESTATION_TIMEOUT_SECONDS
                    ),
                    final_desired_status=(
                        cast(str, desired)
                        if desired in {"RUNNING", "EXITED", "TERMINATED"}
                        else None
                    ),
                    expected_gpu_id=offer.gpu_type_id,
                    observed_gpu_id=(
                        gpu_id if gpu_id == offer.gpu_type_id else None
                    ),
                    gpu_count=gpu_count,
                    cost_attestation=None,
                    observed_gpu_id_sha256=(
                        hashlib.sha256(gpu_id.encode("utf-8")).hexdigest()
                        if gpu_id is not None and gpu_id != offer.gpu_type_id
                        else None
                    ),
                )
            )
            _require(
                get_type in {"boolean", "missing", "null", "string"},
                "pod_interruptible_type_invalid",
            )
            if marker is not _MISSING and marker is not None:
                _require(
                    isinstance(marker, str)
                    and hmac.compare_digest(marker, request.run_marker),
                    "pod_gpu_attestation_marker_mismatch",
                )
            if get_value is True:
                raise RunPodAPIError("pod_interruptible_true")
            if desired is not _MISSING and desired is not None:
                _require(desired == "RUNNING", "pod_gpu_attestation_status_mismatch")
            machine_value = verified.get("machine", _MISSING)
            if isinstance(machine_value, Mapping):
                secure_cloud = machine_value.get("secureCloud", _MISSING)
                if secure_cloud is not _MISSING and secure_cloud is not None:
                    _require(
                        isinstance(secure_cloud, bool)
                        and secure_cloud == (offer.cloud_type == "SECURE"),
                        "pod_gpu_attestation_cloud_type_mismatch",
                    )
            if gpu_id is not None:
                _require(gpu_id == offer.gpu_type_id, "pod_gpu_attestation_gpu_mismatch")
            if gpu_count is not None:
                _require(gpu_count == 1, "pod_gpu_attestation_gpu_count_mismatch")

            observed_price = self._optional_attested_price(
                verified,
                offer=offer,
                maximum=request.hourly_cost_usd,
            )
            if create_price is not None and observed_price is not None:
                _require(
                    abs(create_price - observed_price)
                    <= ON_DEMAND_PRICE_TOLERANCE_USD,
                    "pod_gpu_attestation_price_mismatch",
                )
            actual_price = observed_price if observed_price is not None else create_price

            inventory = self._attestation_inventory(deadline)
            receipt_bound = tuple(
                pod
                for pod in inventory.pods
                if hmac.compare_digest(pod.pod_id, created.pod_id)
            )
            unexpected = tuple(
                pod
                for pod in inventory.pods
                if not hmac.compare_digest(pod.pod_id, created.pod_id)
            )
            marker_matches = bool(receipt_bound) and all(
                pod.run_marker is None
                or hmac.compare_digest(pod.run_marker, request.run_marker)
                for pod in receipt_bound
            )
            receipt_bound_match = (
                (self._receipt_binding_confirmed or not self._require_receipt_binding)
                and len(receipt_bound) == 1
                and marker_matches
            )
            inventory_failure: str | None = None
            if unexpected or len(receipt_bound) > 1 or (
                receipt_bound and not marker_matches
            ):
                inventory_failure = "UNEXPECTED_POD_INVENTORY"
            elif (
                inventory.endpoint_ids
                or inventory.network_volume_ids
                or inventory.template_ids
            ):
                inventory_failure = "pod_attestation_related_resource_present"

            ready = (
                isinstance(marker, str)
                and hmac.compare_digest(marker, request.run_marker)
                and desired == "RUNNING"
                and gpu_id is not None
                and gpu_path is not None
                and gpu_count == 1
                and actual_price is not None
                and receipt_bound_match
            )
            elapsed = max(0.0, self._monotonic() - started)
            cost_attestation = (
                "graphql_uninterruptable_price_match"
                if actual_price is not None
                else None
            )
            self._record_gpu_progress(
                PodGPUAttestationProgressDiagnostic(
                    outcome="pending",
                    failure_code=None,
                    normalized_gpu_path=gpu_path,
                    poll_count=poll_count,
                    poll_elapsed_seconds=min(
                        elapsed, POD_GPU_ATTESTATION_TIMEOUT_SECONDS
                    ),
                    final_desired_status=(
                        cast(str, desired)
                        if desired in {"RUNNING", "EXITED", "TERMINATED"}
                        else None
                    ),
                    expected_gpu_id=offer.gpu_type_id,
                    observed_gpu_id=(
                        gpu_id if gpu_id == offer.gpu_type_id else None
                    ),
                    gpu_count=gpu_count,
                    cost_attestation=cost_attestation,
                    receipt_bound_pod_count=len(receipt_bound),
                    unexpected_pod_count=len(unexpected),
                    endpoint_count=len(inventory.endpoint_ids),
                    network_volume_count=len(inventory.network_volume_ids),
                    template_count=len(inventory.template_ids),
                    receipt_bound_match=receipt_bound_match,
                    observed_gpu_id_sha256=(
                        hashlib.sha256(gpu_id.encode("utf-8")).hexdigest()
                        if gpu_id is not None and gpu_id != offer.gpu_type_id
                        else None
                    ),
                )
            )
            if inventory_failure is not None:
                raise RunPodAPIError(inventory_failure)
            if elapsed >= self._attestation_timeout_seconds:
                current = self._last_gpu_attestation_progress
                if current is not None:
                    self._last_gpu_attestation_progress = replace(
                        current,
                        poll_elapsed_seconds=self._attestation_timeout_seconds,
                    )
                raise RunPodAPIError("POD_GPU_ATTESTATION_TIMEOUT")
            if ready:
                attested_price = cast(Decimal, actual_price)
                attested_gpu_path = cast(str, gpu_path)
                attested_gpu_id = cast(str, gpu_id)
                price_delta = abs(attested_price - offer.hourly_price)
                evidence = (
                    "explicit_interruptible_false"
                    if create_value is False or get_value is False
                    else "request_and_on_demand_price_attested"
                )
                false_source = (
                    "create_response"
                    if create_value is False
                    else "authenticated_get"
                    if get_value is False
                    else None
                )
                attestation = self._rental_attestation(
                    evidence=evidence,
                    request_interruptible=False,
                    offer=offer,
                    actual_price=attested_price,
                    price_delta=price_delta,
                    create_present=create_present,
                    create_type=create_type,
                    get_status=200,
                    get_present=get_present,
                    get_type=get_type,
                    pod_inventory_count=1,
                    explicit_false_source=false_source,
                    normalized_gpu_path=attested_gpu_path,
                    poll_count=poll_count,
                    poll_elapsed_seconds=elapsed,
                    observed_gpu_id=attested_gpu_id,
                    gpu_count=1,
                )
                return attestation
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                current = self._last_gpu_attestation_progress
                if current is not None:
                    self._last_gpu_attestation_progress = replace(
                        current,
                        poll_elapsed_seconds=self._attestation_timeout_seconds,
                    )
                raise RunPodAPIError("POD_GPU_ATTESTATION_TIMEOUT")
            self._sleep(min(self._attestation_poll_seconds, remaining))

    def _record_gpu_progress(
        self,
        diagnostic: PodGPUAttestationProgressDiagnostic,
    ) -> None:
        self._last_gpu_attestation_progress = diagnostic
        if self._record_gpu_attestation_progress is not None:
            self._record_gpu_attestation_progress(diagnostic)

    def _fail_gpu_attestation_progress(self, failure_code: str) -> None:
        current = self._last_gpu_attestation_progress
        if current is None or current.outcome == "failed":
            return
        failed = replace(current, outcome="failed", failure_code=failure_code)
        self._last_gpu_attestation_progress = failed
        if self._record_gpu_attestation_progress is not None:
            with suppress(BaseException):
                # Failure telemetry must never mask the receipt-bound cleanup carrier.
                self._record_gpu_attestation_progress(failed)

    def _attestation_inventory(self, deadline: float) -> RunPodInventory:
        pod_rows = self._get_collection(
            "pods",
            params={"includeMachine": "true", "includeNetworkVolume": "true"},
            timeout_seconds=self._remaining_attestation_seconds(deadline),
        )
        pods = self._pod_records(
            pod_rows,
            duplicate_code="UNEXPECTED_POD_INVENTORY",
        )
        endpoint_ids = self._attestation_ids("endpoints", "endpoint_id_invalid", deadline)
        network_volume_ids = self._attestation_ids(
            "networkvolumes",
            "network_volume_id_invalid",
            deadline,
        )
        template_ids = self._attestation_ids("templates", "template_id_invalid", deadline)
        return RunPodInventory(
            pods=pods,
            endpoint_ids=endpoint_ids,
            network_volume_ids=network_volume_ids,
            template_ids=template_ids,
        )

    def _attestation_ids(
        self,
        path: str,
        code: str,
        deadline: float,
    ) -> tuple[str, ...]:
        rows = self._get_collection(
            path,
            timeout_seconds=self._remaining_attestation_seconds(deadline),
        )
        return tuple(sorted(_resource_id(row.get("id"), code) for row in rows))

    def _remaining_attestation_seconds(self, deadline: float) -> float:
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            raise RunPodAPIError("POD_GPU_ATTESTATION_TIMEOUT")
        return min(remaining, 30.0)

    @staticmethod
    def _optional_attested_price(
        row: Mapping[str, object],
        *,
        offer: GPUOffer,
        maximum: Decimal,
    ) -> Decimal | None:
        value = row.get("costPerHr", _MISSING)
        if value is _MISSING or value is None:
            return None
        price = _decimal(value, "pod_gpu_attestation_price_invalid")
        _require(
            Decimal("0") < price <= maximum,
            "pod_gpu_attestation_price_exceeded",
        )
        _require(
            abs(price - offer.hourly_price) <= ON_DEMAND_PRICE_TOLERANCE_USD,
            "pod_gpu_attestation_price_mismatch",
        )
        return price

    @staticmethod
    def _normalized_gpu(
        row: Mapping[str, object],
    ) -> tuple[str | None, str | None, int | None]:
        ids: list[tuple[str, str]] = []
        counts: list[int] = []
        gpu_value = row.get("gpu", _MISSING)
        if gpu_value is not _MISSING and gpu_value is not None:
            gpu = _object(gpu_value, "pod_gpu_attestation_gpu_invalid")
            gpu_id = gpu.get("id", _MISSING)
            if gpu_id is not _MISSING and gpu_id is not None:
                _require(
                    isinstance(gpu_id, str) and bool(_GPU_TYPE_ID.fullmatch(gpu_id)),
                    "pod_gpu_attestation_gpu_invalid",
                )
                ids.append(("gpu.id", cast(str, gpu_id)))
            gpu_count = gpu.get("count", _MISSING)
            if gpu_count is not _MISSING and gpu_count is not None:
                _require(
                    isinstance(gpu_count, int) and not isinstance(gpu_count, bool),
                    "pod_gpu_attestation_gpu_count_invalid",
                )
                counts.append(cast(int, gpu_count))
        machine_value = row.get("machine", _MISSING)
        if machine_value is not _MISSING and machine_value is not None:
            machine = _object(machine_value, "pod_gpu_attestation_machine_invalid")
            machine_gpu_id = machine.get("gpuTypeId", _MISSING)
            if machine_gpu_id is not _MISSING and machine_gpu_id is not None:
                _require(
                    isinstance(machine_gpu_id, str)
                    and bool(_GPU_TYPE_ID.fullmatch(machine_gpu_id)),
                    "pod_gpu_attestation_gpu_invalid",
                )
                ids.append(("machine.gpuTypeId", cast(str, machine_gpu_id)))
            gpu_type_value = machine.get("gpuType", _MISSING)
            if gpu_type_value is not _MISSING and gpu_type_value is not None:
                gpu_type = _object(
                    gpu_type_value,
                    "pod_gpu_attestation_gpu_invalid",
                )
                nested_id = gpu_type.get("id", _MISSING)
                if nested_id is not _MISSING and nested_id is not None:
                    _require(
                        isinstance(nested_id, str)
                        and bool(_GPU_TYPE_ID.fullmatch(nested_id)),
                        "pod_gpu_attestation_gpu_invalid",
                    )
                    ids.append(("machine.gpuType.id", cast(str, nested_id)))
                nested_count = gpu_type.get("count", _MISSING)
                if nested_count is not _MISSING and nested_count is not None:
                    _require(
                        isinstance(nested_count, int)
                        and not isinstance(nested_count, bool),
                        "pod_gpu_attestation_gpu_count_invalid",
                    )
                    counts.append(cast(int, nested_count))
        if ids:
            _require(
                len({gpu_id for _path, gpu_id in ids}) == 1,
                "pod_gpu_attestation_gpu_mismatch",
            )
        if counts:
            _require(
                len(set(counts)) == 1,
                "pod_gpu_attestation_gpu_count_mismatch",
            )
        path, gpu_id = ids[0] if ids else (None, None)
        return gpu_id, path, counts[0] if counts else None

    def _rental_attestation(
        self,
        *,
        evidence: str,
        request_interruptible: bool,
        offer: GPUOffer,
        actual_price: Decimal,
        price_delta: Decimal,
        create_present: bool,
        create_type: str,
        get_status: int | None,
        get_present: bool | None,
        get_type: str | None,
        pod_inventory_count: int | None,
        explicit_false_source: str | None,
        normalized_gpu_path: str,
        poll_count: int,
        poll_elapsed_seconds: float,
        observed_gpu_id: str,
        gpu_count: int,
    ) -> PodRentalAttestationDiagnostic:
        return PodRentalAttestationDiagnostic(
            evidence=evidence,
            request_interruptible=request_interruptible,
            create_http_status=201,
            selected_gpu_id=offer.gpu_type_id,
            selected_uninterruptable_price=offer.hourly_price,
            create_cost_per_hr=actual_price,
            price_delta_usd=price_delta,
            desired_status="RUNNING",
            cloud_type=offer.cloud_type,
            create_interruptible_present=create_present,
            create_interruptible_json_type=create_type,
            get_verification_http_status=get_status,
            get_interruptible_present=get_present,
            get_interruptible_json_type=get_type,
            pod_inventory_count=pod_inventory_count,
            explicit_false_source=explicit_false_source,
            create_http_class="success_201",
            normalized_gpu_path=normalized_gpu_path,
            gpu_poll_count=poll_count,
            gpu_poll_elapsed_seconds=poll_elapsed_seconds,
            observed_gpu_id=observed_gpu_id,
            gpu_count=gpu_count,
            cost_attestation="graphql_uninterruptable_price_match",
        )

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

    def await_pod_connectivity(
        self,
        pod: PodRecord,
        request: PodRequest,
        *,
        ssh_probe: Callable[[PodConnection, float], bool],
    ) -> PodConnection:
        """Await IP, SSH port mapping, and an actual SSH probe on one bound Pod."""

        safe_id = _resource_id(pod.pod_id, "POD_CONNECTIVITY_POD_ID_INVALID")
        _require(pod.run_marker == request.run_marker, "POD_CONNECTIVITY_MARKER_MISMATCH")
        offer = self._last_offer
        attestation = self._last_rental_attestation
        if (
            offer is None
            or attestation is None
            or request.gpu_type_id != offer.gpu_type_id
        ):
            raise RunPodAPIError("POD_CONNECTIVITY_ALLOCATION_NOT_ATTESTED")
        started = self._monotonic()
        deadline = started + self._connectivity_timeout_seconds
        poll_count = 0
        ssh_attempted = False
        self._record_connectivity(
            PodConnectivityProgressDiagnostic(
                outcome="pending",
                failure_code=None,
                public_ip_present=False,
                tcp_port_present=False,
                poll_count=0,
                elapsed_seconds=0.0,
                ssh_ready=False,
            )
        )
        try:
            while True:
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    raise RunPodAPIError(
                        "POD_SSH_READINESS_TIMEOUT"
                        if ssh_attempted
                        else "POD_CONNECTIVITY_TIMEOUT"
                    )
                poll_count += 1
                try:
                    payload = self._request_json(
                        "GET",
                        f"pods/{safe_id}",
                        expected_status=200,
                        params={
                            "includeMachine": "true",
                            "includeNetworkVolume": "true",
                        },
                        timeout_seconds=min(remaining, 30.0),
                    )
                except RunPodAPIError as exc:
                    if exc.code == "unexpected_status_404":
                        raise RunPodAPIError("POD_CONNECTIVITY_POD_MISSING") from None
                    raise
                row = _object(payload, "POD_CONNECTIVITY_RESPONSE_INVALID")
                _require(row.get("id") == safe_id, "POD_CONNECTIVITY_ID_MISMATCH")
                marker = row.get("name", _MISSING)
                if marker is not _MISSING and marker is not None:
                    _require(
                        marker == request.run_marker,
                        "POD_CONNECTIVITY_MARKER_MISMATCH",
                    )

                desired = row.get("desiredStatus", _MISSING)
                if desired in {"FAILED", "EXITED", "TERMINATED"}:
                    raise RunPodAPIError(f"POD_CONNECTIVITY_TERMINAL_{desired}")
                if desired is not _MISSING and desired is not None:
                    _require(desired == "RUNNING", "POD_CONNECTIVITY_STATUS_INVALID")

                interruptible = row.get("interruptible", _MISSING)
                _require(
                    _json_type(interruptible) in {"boolean", "missing", "null", "string"},
                    "POD_CONNECTIVITY_INTERRUPTIBLE_INVALID",
                )
                if interruptible is True:
                    raise RunPodAPIError("POD_CONNECTIVITY_INTERRUPTIBLE_TRUE")

                gpu_id, _gpu_path, gpu_count = self._normalized_gpu(row)
                if gpu_id is not None:
                    _require(gpu_id == offer.gpu_type_id, "POD_CONNECTIVITY_GPU_MISMATCH")
                if gpu_count is not None:
                    _require(gpu_count == 1, "POD_CONNECTIVITY_GPU_COUNT_MISMATCH")

                machine_value = row.get("machine", _MISSING)
                if isinstance(machine_value, Mapping):
                    secure_cloud = machine_value.get("secureCloud", _MISSING)
                    if secure_cloud is not _MISSING and secure_cloud is not None:
                        _require(
                            isinstance(secure_cloud, bool)
                            and secure_cloud == (offer.cloud_type == "SECURE"),
                            "POD_CONNECTIVITY_CLOUD_TYPE_MISMATCH",
                        )

                price_value = row.get("costPerHr", _MISSING)
                if price_value is not _MISSING and price_value is not None:
                    price = _decimal(price_value, "POD_CONNECTIVITY_PRICE_INVALID")
                    _require(
                        Decimal("0") < price <= request.hourly_cost_usd
                        and abs(price - offer.hourly_price)
                        <= ON_DEMAND_PRICE_TOLERANCE_USD,
                        "POD_CONNECTIVITY_PRICE_MISMATCH",
                    )
                else:
                    price = attestation.create_cost_per_hr

                ip_value = row.get("publicIp", _MISSING)
                public_ip_present = isinstance(ip_value, str) and bool(ip_value.strip())
                public_ip: str | None = None
                if public_ip_present:
                    try:
                        public_ip = str(ipaddress.ip_address(cast(str, ip_value).strip()))
                    except ValueError as exc:
                        raise RunPodAPIError("POD_CONNECTIVITY_PUBLIC_IP_INVALID") from exc
                elif (
                    ip_value is not _MISSING
                    and ip_value is not None
                    and not (isinstance(ip_value, str) and not ip_value.strip())
                ):
                    raise RunPodAPIError("POD_CONNECTIVITY_PUBLIC_IP_INVALID")

                ports_value = row.get("ports", _MISSING)
                if ports_value is not _MISSING and ports_value is not None:
                    ports = _sequence(ports_value, "POD_CONNECTIVITY_PORTS_INVALID")
                    _require(set(ports) == {"22/tcp"}, "POD_CONNECTIVITY_PORTS_INVALID")
                mappings_value = row.get("portMappings", _MISSING)
                public_port: int | None = None
                if mappings_value is not _MISSING and mappings_value is not None:
                    mappings = _object(
                        mappings_value,
                        "POD_CONNECTIVITY_TCP_PORT_INVALID",
                    )
                    mapped = mappings.get("22", _MISSING)
                    if mapped is not _MISSING and mapped is not None:
                        _require(
                            isinstance(mapped, int)
                            and not isinstance(mapped, bool)
                            and 1 <= mapped <= 65_535,
                            "POD_CONNECTIVITY_TCP_PORT_INVALID",
                        )
                        public_port = cast(int, mapped)
                tcp_port_present = public_port is not None
                allocation_safe = (
                    desired == "RUNNING"
                    and gpu_id == offer.gpu_type_id
                    and gpu_count == 1
                )
                ssh_ready = False
                connection: PodConnection | None = None
                if (
                    allocation_safe
                    and public_ip is not None
                    and public_port is not None
                ):
                    connection = PodConnection(
                        pod_id=safe_id,
                        public_ip=public_ip,
                        public_ssh_port=public_port,
                        gpu_display_name=offer.display_name,
                        hourly_price=price,
                    )
                    ssh_attempted = True
                    try:
                        ssh_ready = bool(
                            ssh_probe(connection, min(max(remaining, 0.001), 15.0))
                        )
                    except Exception:
                        ssh_ready = False

                elapsed = min(
                    max(0.0, self._monotonic() - started),
                    self._connectivity_timeout_seconds,
                )
                progress = PodConnectivityProgressDiagnostic(
                    outcome="ready" if ssh_ready else "pending",
                    failure_code=None,
                    public_ip_present=public_ip_present,
                    tcp_port_present=tcp_port_present,
                    poll_count=poll_count,
                    elapsed_seconds=elapsed,
                    ssh_ready=ssh_ready,
                )
                self._record_connectivity(progress)
                if ssh_ready and connection is not None:
                    return connection

                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    raise RunPodAPIError(
                        "POD_SSH_READINESS_TIMEOUT"
                        if ssh_attempted
                        else "POD_CONNECTIVITY_TIMEOUT"
                    )
                poll_seconds = (
                    POD_CONNECTIVITY_FAST_POLL_SECONDS
                    if elapsed < POD_CONNECTIVITY_FAST_POLL_WINDOW_SECONDS
                    else POD_CONNECTIVITY_SLOW_POLL_SECONDS
                )
                self._sleep(min(poll_seconds, remaining))
        except RunPodAPIError as exc:
            self._fail_connectivity_progress(exc.code)
            raise

    def _record_connectivity(
        self,
        diagnostic: PodConnectivityProgressDiagnostic,
    ) -> None:
        self._last_connectivity_progress = diagnostic
        if self._record_connectivity_progress is not None:
            self._record_connectivity_progress(diagnostic)

    def _fail_connectivity_progress(self, failure_code: str) -> None:
        current = self._last_connectivity_progress
        if current is None or current.outcome == "failed":
            return
        failed = replace(
            current,
            outcome="failed",
            failure_code=failure_code,
            elapsed_seconds=(
                self._connectivity_timeout_seconds
                if failure_code
                in {"POD_CONNECTIVITY_TIMEOUT", "POD_SSH_READINESS_TIMEOUT"}
                else current.elapsed_seconds
            ),
        )
        self._last_connectivity_progress = failed
        if self._record_connectivity_progress is not None:
            with suppress(BaseException):
                self._record_connectivity_progress(failed)

    def _list_ids(self, path: str, code: str) -> tuple[str, ...]:
        rows = self._get_collection(path)
        return tuple(sorted(_resource_id(row.get("id"), code) for row in rows))

    def _get_collection(
        self,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> tuple[Mapping[str, object], ...]:
        payload = self._request_json(
            "GET",
            path,
            expected_status=200,
            params=params,
            timeout_seconds=timeout_seconds,
        )
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
        response_observer: Callable[[int, bytearray, str | None], None] | None = None,
        status_classifier: Callable[[int], str] | None = None,
        timeout_seconds: float | None = None,
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
                timeout=(
                    self._client.timeout
                    if timeout_seconds is None
                    else httpx.Timeout(timeout_seconds)
                ),
            ) as response:
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_RESPONSE_BYTES:
                        raise RunPodAPIError("response_too_large")
                status = response.status_code
                content_type = response.headers.get("content-type")
        except RunPodAPIError:
            raise
        except httpx.HTTPError:
            raise RunPodAPIError("transport_failed") from None
        if response_observer is not None:
            response_observer(status, body, content_type)
        if status != expected_status:
            code = (
                f"unexpected_status_{status}"
                if status_classifier is None
                else status_classifier(status)
            )
            raise RunPodAPIError(code)
        if expected_status == 204:
            _require(not body, "delete_response_not_empty")
            return None
        _require(bool(body), "response_body_missing")
        try:
            return json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RunPodAPIError("response_json_invalid") from None

    def _classify_create_status(self, status: int) -> str:
        if status == 400:
            diagnostic = self._last_create_response_diagnostic
            if (
                diagnostic is not None
                and diagnostic.http_status == status
                and diagnostic.failure_code
                in {
                    "GPU_CAPACITY_ALLOCATION_REJECTED",
                    "RUNPOD_CREATE_PAYLOAD_INVALID",
                    "RUNPOD_CREATE_BAD_REQUEST_UNKNOWN",
                }
            ):
                return diagnostic.failure_code
            return "RUNPOD_CREATE_BAD_REQUEST_UNKNOWN"
        if status in _ALLOCATION_ERROR_STATUSES:
            return "GPU_CAPACITY_ALLOCATION_REJECTED"
        if status == 401:
            return "RUNPOD_AUTH_INVALID"
        if status == 403:
            return "RUNPOD_PERMISSION_DENIED"
        if status == 429:
            return "RUNPOD_RATE_LIMITED"
        if 500 <= status <= 599:
            return "RUNPOD_PROVIDER_ERROR"
        return "RUNPOD_CREATE_RESPONSE_REJECTED"

    def _observe_create_response(
        self,
        status: int,
        body: bytearray,
        content_type_header: str | None,
    ) -> None:
        byte_length = len(body)
        content_type = _safe_content_type(content_type_header)
        digest = hashlib.sha256(body).hexdigest()
        if not body:
            self._last_create_response_diagnostic = CreateResponseDiagnostic(
                http_status=status,
                top_level_keys=(),
                id_present=None,
                interruptible_present=None,
                interruptible_json_type="unavailable",
                classification=(
                    "http_201_empty" if status == 201 else "provider_error_empty"
                ),
                body_kind="empty",
                byte_length=0,
                sha256=digest,
                content_type=content_type,
                failure_code=self._bad_request_code(status, ""),
            )
            return
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            try:
                error_text: object = bytes(body).decode("utf-8", errors="strict")
                body_kind = "non_json_text"
            except UnicodeDecodeError:
                error_text = ""
                body_kind = "non_json_binary"
            self._last_create_response_diagnostic = CreateResponseDiagnostic(
                http_status=status,
                top_level_keys=(),
                id_present=None,
                interruptible_present=None,
                interruptible_json_type="unavailable",
                classification=(
                    "http_201_non_json" if status == 201 else "provider_error_non_json"
                ),
                body_kind=body_kind,
                byte_length=byte_length,
                sha256=digest,
                content_type=content_type,
                failure_code=self._bad_request_code(status, error_text),
            )
            return
        if isinstance(payload, Mapping):
            row = cast(Mapping[str, object], payload)
            keys = tuple(
                sorted(
                    key
                    if _SAFE_JSON_KEY.fullmatch(key)
                    and not _SENSITIVE_JSON_KEY.search(key)
                    else "<redacted-key>"
                    for key in row
                )
            )
            interruptible = row.get("interruptible", _MISSING)
            self._last_create_response_diagnostic = CreateResponseDiagnostic(
                http_status=status,
                top_level_keys=keys,
                id_present="id" in row,
                interruptible_present="interruptible" in row,
                interruptible_json_type=_json_type(interruptible),
                classification=(
                    "http_201_create_success_object"
                    if status == 201
                    else "provider_error_json_object"
                ),
                body_kind="json_object",
                byte_length=byte_length,
                sha256=None,
                content_type=content_type,
                failure_code=self._bad_request_code(status, row),
            )
            return
        validation_errors = self._validation_errors(payload)
        body_json_type = _json_type(payload)
        body_kind = (
            "json_array" if body_json_type == "array" else f"json_scalar_{body_json_type}"
        )
        self._last_create_response_diagnostic = CreateResponseDiagnostic(
            http_status=status,
            top_level_keys=(),
            id_present=None,
            interruptible_present=None,
            interruptible_json_type="unavailable",
            classification=(
                f"http_201_{body_kind}"
                if status == 201
                else f"provider_error_{body_kind}"
            ),
            body_kind=body_kind,
            byte_length=byte_length,
            sha256=digest if body_json_type != "array" else None,
            content_type=content_type,
            failure_code=self._bad_request_code(
                status,
                payload,
                validation_errors=validation_errors,
            ),
            validation_errors=validation_errors,
        )

    def _bad_request_code(
        self,
        status: int,
        payload: object,
        *,
        validation_errors: tuple[CreateValidationErrorDiagnostic, ...] = (),
    ) -> str | None:
        if status != 400:
            return None
        texts = self._error_texts(payload)
        if any(_CAPACITY_CODE.fullmatch(text) for text in texts) or any(
            _CAPACITY_MESSAGE.search(text) for text in texts
        ):
            return "GPU_CAPACITY_ALLOCATION_REJECTED"
        if validation_errors or any(_VALIDATION_MESSAGE.search(text) for text in texts):
            return "RUNPOD_CREATE_PAYLOAD_INVALID"
        return "RUNPOD_CREATE_BAD_REQUEST_UNKNOWN"

    def _error_texts(self, value: object) -> tuple[str, ...]:
        collected: list[str] = []

        def visit(item: object, *, key: str | None = None) -> None:
            if len(collected) >= 32:
                return
            if isinstance(item, Mapping):
                for raw_key, child in item.items():
                    candidate = str(raw_key)
                    if _SENSITIVE_JSON_KEY.search(candidate):
                        continue
                    if candidate.lower() in {
                        "code",
                        "type",
                        "message",
                        "msg",
                        "detail",
                        "error",
                        "reason",
                    }:
                        visit(child, key=candidate)
                return
            if isinstance(item, Sequence) and not isinstance(
                item, str | bytes | bytearray
            ):
                for child in item[:32]:
                    visit(child, key=key)
                return
            if isinstance(item, str) and key is not None and len(item) <= 512:
                collected.append(item)

        if isinstance(value, str):
            if len(value) <= 512:
                collected.append(value)
        else:
            visit(value)
        return tuple(collected)

    def _validation_errors(
        self,
        payload: object,
    ) -> tuple[CreateValidationErrorDiagnostic, ...]:
        if not isinstance(payload, list):
            return ()
        errors: list[CreateValidationErrorDiagnostic] = []
        for item in payload[:32]:
            if not isinstance(item, Mapping):
                continue
            row = cast(Mapping[str, object], item)
            location = row.get("loc", row.get("field", row.get("path", _MISSING)))
            raw_code = row.get("type", row.get("code", _MISSING))
            message = row.get("msg", row.get("message", _MISSING))
            if location is _MISSING and raw_code is _MISSING:
                continue
            code = _safe_error_code(raw_code)
            errors.append(
                CreateValidationErrorDiagnostic(
                    field_path=_safe_field_path(location),
                    type_code=code,
                    message_class=_validation_message_class(code, message),
                )
            )
        return tuple(errors)


__all__ = [
    "CreatePayloadContractReport",
    "CreateResponseDiagnostic",
    "CreateValidationErrorDiagnostic",
    "GPUAvailabilityReport",
    "GPUCandidateDiagnostic",
    "GPUOffer",
    "GraphQLErrorDiagnostic",
    "GRAPHQL_URL",
    "MAPILLARY_ENV_REFERENCE",
    "MAX_CONTAINER_DISK_GB",
    "MIN_GPU_MEMORY_GB",
    "ON_DEMAND_PRICE_TOLERANCE_USD",
    "POD_CONNECTIVITY_FAST_POLL_SECONDS",
    "POD_CONNECTIVITY_FAST_POLL_WINDOW_SECONDS",
    "POD_CONNECTIVITY_SLOW_POLL_SECONDS",
    "POD_CONNECTIVITY_TIMEOUT_SECONDS",
    "POD_GPU_ATTESTATION_POLL_SECONDS",
    "POD_GPU_ATTESTATION_TIMEOUT_SECONDS",
    "POD_VOLUME_GB",
    "PodConnection",
    "PodConnectivityProgressDiagnostic",
    "PodGPUAttestationProgressDiagnostic",
    "PodRentalAttestationDiagnostic",
    "REST_BASE_URL",
    "RunPodAPIError",
    "RunPodBillingSnapshot",
    "RunPodCreateError",
    "RunPodConfig",
    "RunPodInventory",
    "RunPodV1Client",
    "build_create_payload",
    "validate_create_payload_contract",
]
