# Phase 3F RunPod operator handoff

This handoff controls one bounded, private Phase 3F Pod. The start command is a
local-only dry-run unless `-Execute` is present. The API key is read only from
the PowerShell process environment; it is not accepted as an argument or
printed. The wrapper does not read `.env` or `asda.html`.

## Commands

Dry-run from any PowerShell working directory:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\geoSearch\scripts\start-phase3f-runpod.ps1" -MaxSpendUsd 10 -SoftStopUsd 7.5 -HardStopUsd 9 -MaxGpuHourlyUsd 0.50 -MaxWallMinutes 345
```

The dry-run performs no RunPod request and now renders only the create payload's
field names and JSON types. Require `payload_contract_valid=true`,
`create_attempts=0`, `cloud_mutations=0`, and
`secret_values_included=false`. Environment values, including the RunPod secret
reference and ephemeral SSH public key, are never rendered.

Authenticated read-only readiness immediately before any paid retry:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\geoSearch\scripts\start-phase3f-runpod.ps1" -LiveReadiness -RuntimeRoot "C:\AtlasLensRuntime\phase3f" -MaxSpendUsd 10 -SoftStopUsd 7.5 -HardStopUsd 9 -MaxGpuHourlyUsd 0.50 -MaxWallMinutes 345
```

This mode verifies all four inventories, discovers current provider GPU IDs,
queries only discovered candidates with at least 16 GiB, and reports sanitized
stock, price, capacity evidence and rejection classifications. A list-valued
`availableGpuCounts` must contain `1`. A null count list is accepted only as
`capacity_unconfirmed_but_advertised` when stock is High, Medium or Low and the
remaining memory, cloud and price gates pass. It cannot create or terminate a
resource. Do not run the paid command unless it returns
`ready_for_execute=true`, `cloud_mutations=0`, and inventories `0/0/0/0`.

Real start, only after reviewing the dry-run result:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\geoSearch\scripts\start-phase3f-runpod.ps1" -Execute -MaxSpendUsd 10 -SoftStopUsd 7.5 -HardStopUsd 9 -MaxGpuHourlyUsd 0.50 -MaxWallMinutes 345
```

Status from a separate PowerShell session:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\geoSearch\scripts\status-phase3f-runpod.ps1"
```

Emergency termination:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\geoSearch\scripts\stop-phase3f-runpod.ps1"
```

The stop wrapper reads only
`C:\AtlasLensRuntime\phase3f\_operator\phase3f-current.json`, requires the
receipt Pod ID and run marker to agree with inventory, calls terminate rather
than stop, and refuses to mutate when any unrelated Pod or other RunPod
resource is present.

## Fail-closed lifecycle

Immediately before creation the supervisor requires Pod, endpoint, network
volume and template inventories to all be empty. It makes one create attempt,
never retries with a second Pod, and revalidates the exact selected GPU ID with
one read-only detail query. Selection uses A5000, L4, RTX 3090, then other
eligible GPUs; price breaks ties only within the same preference level. Secure
Cloud is preferred whenever its variant is eligible, with Community Cloud used
only when no eligible Secure variant exists. Offers must remain at or below the
configured hourly ceiling. An advertised-but-unconfirmed offer may make exactly
one REST create attempt. A sanitized allocation/no-capacity response becomes
`GPU_CAPACITY_RACE_NO_POD`; no alternate GPU or Pod is tried, no successful-run
receipt or spend is recorded, and all four inventories must be restored to zero.
Create response handling classifies HTTP status before looking at any Pod field.
Only HTTP 201 enters success parsing. HTTP 400 is a capacity race only when a
strict provider code/message allowlist explicitly reports no instances or
capacity; the pre-existing 404/409/422 allocation statuses remain capacity
failures. HTTP 400 schema/field validation is
`RUNPOD_CREATE_PAYLOAD_INVALID`; an unrecognized bad request is
`RUNPOD_CREATE_BAD_REQUEST_UNKNOWN`. JSON-array validation diagnostics retain
only the safe field path, type/code, and a general message class. Text/scalar or
empty errors retain only byte length, SHA-256, normalized content type, and an
allowlist classification; raw response bodies are not persisted or printed. 401
is `RUNPOD_AUTH_INVALID`, 403 is `RUNPOD_PERMISSION_DENIED`, 429 is
`RUNPOD_RATE_LIMITED`, and 5xx is `RUNPOD_PROVIDER_ERROR`. On HTTP 201 the Pod ID
is atomically bound to the operator receipt before any remaining field is
validated. The response must not advertise `interruptible=true`. Exact JSON
boolean `false` in the create response is sufficient; otherwise an authenticated
`GET /pods/{id}` checks the field. Missing, null or string values are never
coerced. If both representations remain indeterminate, the Pod can proceed only
when request `interruptible` is exact boolean `false`, create is HTTP 201, the ID
is already receipt-bound, returned GPU and `RUNNING` status match, `costPerHr` is
positive and within both the hourly ceiling and 0.005 USD/hour of the revalidated
GraphQL `uninterruptablePrice`, neither representation proves boolean `true`, and
inventory shows only the receipt-bound Pod with no endpoint, network volume or
template. This evidence is recorded as
`request_and_on_demand_price_attested`; it is never described as
`interruptible_field_verified`. Receipt v2 also records the atomic binding time
and sanitized field-presence/type evidence while remaining able to read legacy
v1 receipts. Every post-ID validation failure carries the receipt-bound Pod into
the one-Pod `finally` termination path.
Create-time allocation fields are provisional. A HTTP-201 response may contain
`machine` while omitting `gpu`; that omission alone is not a failure. After the
ID is bound, authenticated `GET /pods/{id}` calls use `includeMachine=true` and
a monotonic deadline of at most 180 seconds with a controlled interval. This is
GET-only allocation polling, not a create retry. Exact GPU ID evidence is accepted
only from `gpu.id`, `machine.gpuTypeId`, or `machine.gpuType.id`; assigned count
comes only from `gpu.count` or `machine.gpuType.count`. All simultaneously present
paths must agree, and display names or `machine.minPodGpuCount` are not allocation
evidence. Missing/null machine, GPU, status or public-IP values remain pending.
Success requires the exact bound ID and run name, `RUNNING`, count one, selected
GPU ID, valid on-demand price/rental evidence, a public allocation and only the
bound Pod with no endpoint, network volume or template. A mismatch terminates
immediately; expiration raises `POD_GPU_ATTESTATION_TIMEOUT`. Receipt v2 stores
only allowlisted fields or a hash for mismatching GPU text, plus poll count/time
and outcome, and remains able to read receipts written before these fields.
The create request is validated offline before POST against the bounded official
GPU Pod contract. It contains only name, digest-pinned image, cloud/compute type,
the exact selected GPU ID with custom priority, one non-interruptible GPU, a
40 GiB container disk, a 20 GiB Pod volume mounted at `/workspace`, SSH port and
public-IP support, and necessary environment fields. A Pod volume belongs to the
Pod and persists across its restarts; it is not a separately created network
volume. `networkVolumeId`, template/null placeholders, CPU-only fields and
Serverless fields are omitted.
Mapillary access is mapped through the named RunPod secret reference. Source,
canonical-LF vendor files and the exact model are checksum-verified before
transfer. The cloud job is watched in the foreground, output is downloaded and
checksum-verified, and the one-Pod session terminates in `finally`. The start
wrapper invokes receipt-bound termination again from its own `finally` and
requires all four inventories to return to zero.

Stop on these operator blockers: `ACTIVE_POD_INVENTORY_NOT_ZERO`,
`ACTIVE_ENDPOINT_INVENTORY_NOT_ZERO`, `NETWORK_VOLUME_INVENTORY_NOT_ZERO`,
`TEMPLATE_INVENTORY_NOT_ZERO`, `UNEXPECTED_POD_INVENTORY`,
`OPERATOR_PROCESS_ALREADY_RUNNING`, `STALE_OPERATOR_RECEIPT_REQUIRES_TERMINATE`,
`NO_ELIGIBLE_GPU_OFFER`, `GPU_AVAILABILITY_GRAPHQL_ERRORS`,
`GPU_CAPACITY_RACE_NO_POD`,
`RUNPOD_CREATE_PAYLOAD_INVALID`, `RUNPOD_CREATE_BAD_REQUEST_UNKNOWN`,
`RUNPOD_AUTH_INVALID`, `RUNPOD_PERMISSION_DENIED`, `RUNPOD_RATE_LIMITED`,
`RUNPOD_PROVIDER_ERROR`,
`POD_GPU_ATTESTATION_TIMEOUT`, `POD_GPU_ATTESTATION_POD_MISSING`,
`projected_cost_exceeds_target`,
`soft_stop_budget_reached`, `termination_budget_reached`,
`absolute_budget_reached`, `runtime_limit_reached`,
`PHASE3F_TERMINATION_UNVERIFIED`, or `CLOUD_CLEANUP_UNVERIFIED`.

## Local output

- Operator state/receipt:
  `C:\AtlasLensRuntime\phase3f\_operator\phase3f-current.json`
- Verified cloud result:
  `C:\AtlasLensRuntime\phase3f\<publication-or-inventory-sha256>\`
- Sanitized local supervisor receipt:
  `C:\AtlasLensRuntime\phase3f\_receipts\<publication-or-inventory-sha256>.json`

Return the complete console output plus the operator receipt, sanitized local
supervisor receipt, cloud `execution-receipt.json`, and cloud
`checksum-inventory.json` to the follow-up task. Do not send credentials.
