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

## Existing manually managed Pod: resumable Mapillary job

This path is separate from the RunPod supervisor above. It calls no RunPod API,
does not start, stop, terminate, or inspect a Pod, and must be used only from a
shell inside an already-running manually managed Pod. A 4.5-hour job deadline
does not stop the Pod or stop billing. The operator remains responsible for the
Pod lifecycle and must copy verified outputs off the Pod before terminating it.

The failed historical job predates the metadata-page checkpoint and therefore
cannot recover its completed pages. The first repaired run starts with a new
32-hex run ID. If that repaired run is interrupted, later `start-job` calls keep
the same run root, create a fresh output attempt, and resume only when both its
private metadata and cumulative-counter checkpoints are present and valid.

After the repair commit exists locally, create a self-contained Git bundle and
record both roots of trust on Windows. These commands have not been executed on
the stopped Pod:

```powershell
git -C "D:\geoSearch" bundle create "D:\geoSearch\.local\phase3f-mapillary-secret-hotfix.bundle" HEAD
git -C "D:\geoSearch" rev-parse HEAD
(Get-FileHash -Algorithm SHA256 -LiteralPath "D:\geoSearch\.local\phase3f-mapillary-secret-hotfix.bundle").Hash.ToLowerInvariant()
```

Upload that bundle without modifying it to
`/workspace/phase3f-transfer/phase3f-mapillary-secret-hotfix.bundle`. Copy the exact
commit and bundle SHA-256 printed in the final local handoff into the two
variables below. After separately starting the Pod, use these commands inside
the Pod. They are intentionally not Pod-start or RunPod commands and have not
been executed as part of this repair:

`test -n "${MAPILLARY_ACCESS_TOKEN:-}"` proves only that a shell variable has a
value; it does not prove the value is exported to child processes. In the same
shell that received the secret, run `export MAPILLARY_ACCESS_TOKEN` without an
assignment. This does not place the value in argv or print it. Then require the
presence-only status command to emit exactly `RESOLVED_SECRET`. A literal
`{{ RUNPOD_SECRET_... }}` is rejected as
`UNRESOLVED_RUNPOD_SECRET_REFERENCE`; an absent value is rejected as
`MAPILLARY_ACCESS_TOKEN_MISSING`. The value, length, and hash are never emitted.

```bash
set -euo pipefail
expected_commit='<FINAL_COMMIT_SHA>'
expected_bundle_sha256='<FINAL_BUNDLE_SHA256>'
bundle=/workspace/phase3f-transfer/phase3f-mapillary-secret-hotfix.bundle
test "$(sha256sum "$bundle" | awk '{print $1}')" = "$expected_bundle_sha256"
test ! -e /workspace/phase3f-repo-mapillary-resume
git clone --no-checkout "$bundle" /workspace/phase3f-repo-mapillary-resume
git -C /workspace/phase3f-repo-mapillary-resume checkout --detach "$expected_commit"
test "$(git -C /workspace/phase3f-repo-mapillary-resume rev-parse HEAD)" = "$expected_commit"
git -C /workspace/phase3f-repo-mapillary-resume fsck --strict
test -z "$(git -C /workspace/phase3f-repo-mapillary-resume status --porcelain=v1 --untracked-files=all)"
install -d -m 700 /workspace/phase3f-manual
export MAPILLARY_ACCESS_TOKEN
bash /workspace/phase3f-repo-mapillary-resume/scripts/phase3f/existing-pod-secret-status.sh
bash /workspace/phase3f-repo-mapillary-resume/scripts/phase3f/existing-pod-prepare-check.sh \
  --runtime-root /workspace/phase3f-manual \
  --repository-root /workspace/phase3f-repo-mapillary-resume \
  --model /workspace/phase3f-transfer/model.safetensors \
  --vendor-root /workspace/phase3f-transfer/vendor \
  --source-commit "$expected_commit"
bash /workspace/phase3f-repo-mapillary-resume/scripts/phase3f/existing-pod-start-job.sh \
  --runtime-root /workspace/phase3f-manual \
  --repository-root /workspace/phase3f-repo-mapillary-resume \
  --model /workspace/phase3f-transfer/model.safetensors \
  --vendor-root /workspace/phase3f-transfer/vendor \
  --source-commit "$expected_commit"
```

If the repository directory already exists from the preceding immutable bundle,
do not clone over it. Require a clean checkout, verify the new bundle hash, then
run `git fetch "$bundle" HEAD`, detached-checkout the new exact commit, rerun
`git fsck --strict`, and require clean status before the export/status/prepare
commands above.

Status and sanitized log projection, from the already-running Pod:

```bash
bash /workspace/phase3f-repo-mapillary-resume/scripts/phase3f/existing-pod-status-job.sh \
  --runtime-root /workspace/phase3f-manual
bash /workspace/phase3f-repo-mapillary-resume/scripts/phase3f/existing-pod-tail-log.sh \
  --runtime-root /workspace/phase3f-manual --lines 80
```

If the repaired job is interrupted and the Pod remains running, rerun only the
same `existing-pod-start-job.sh` command above; do not rerun `prepare-check`.
That call detects the paired private checkpoints and adds `--resume` without
creating a second job or reusing an output directory. To stop only the
receipt/state-bound in-Pod process group (not the Pod), run:

```bash
bash /workspace/phase3f-repo-mapillary-resume/scripts/phase3f/existing-pod-stop-job.sh \
  --runtime-root /workspace/phase3f-manual
```

Do not terminate the Pod until the selected fresh `output-*` directory and its
checksum inventory have been copied and verified. These helpers never claim or
attempt automatic Pod shutdown.
