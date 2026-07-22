# Phase 3F RunPod operator handoff

This handoff controls one bounded, private Phase 3F Pod. The start command is a
local-only dry-run unless `-Execute` is present. The API key is read only from
the PowerShell process environment; it is not accepted as an argument or
printed. The wrapper does not read `.env` or `asda.html`.

## One-command acquisition to fine-tuning pipeline

The production operator entry point is `scripts\phase3f-end-to-end.ps1`. It
preserves the current local run ID, resumes acquisition only when the current
checkpoint is genuinely unsealed, validates the private corpus, creates
immutable train/validation/locked-holdout manifests,
measures the pinned pretrained MegaLoc baseline on validation, transfers only
the sealed corpus/training package, runs real mixed-precision metric-learning
fine-tuning on one bounded RunPod Pod, retrieves checksummed outputs, terminates
the exact receipt-bound Pod, and verifies the full pre-run cloud inventory is
restored. Execute requires an explicit cloud-consent switch:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\geoSearch\scripts\phase3f-end-to-end.ps1" -Action Preflight
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\geoSearch\scripts\phase3f-end-to-end.ps1" -Action RemoteEnvironmentPlan
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\geoSearch\scripts\phase3f-end-to-end.ps1" -Action TrainingDeadlinePlan -TrainingMaxWallMinutes 345
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\geoSearch\scripts\phase3f-end-to-end.ps1" -Action TrainingMemoryPlan -MaxGpuHourlyUsd 0.50
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\geoSearch\scripts\phase3f-end-to-end.ps1" -Action ReconcileLocalReceipts
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\geoSearch\scripts\phase3f-end-to-end.ps1" -Action CloudPlan
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\geoSearch\scripts\phase3f-end-to-end.ps1" -Action Execute -CloudConsent
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\geoSearch\scripts\phase3f-end-to-end.ps1" -Action Status
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\geoSearch\scripts\phase3f-end-to-end.ps1" -Action Resume -CloudConsent
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\geoSearch\scripts\phase3f-end-to-end.ps1" -Action EmergencyStop
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\geoSearch\scripts\phase3f-end-to-end.ps1" -Action Cleanup
```

`Resume` first runs a read-only local plan. A checksum-valid sealed corpus with
exactly 830 assets and a byte-exact recomputed `READY_FOR_TRAINING` report skips
the Mapillary prompt, acquisition, media resolution, and all dataset writes. A
running or completed training receipt also returns idempotently without starting
that phase again. Missing/tampered seal or readiness evidence fails with the
typed artifact/field blocker before any cloud secret is requested. Only a real
unsealed acquisition checkpoint requests the Mapillary token.

`CloudPlan` is the mutation-free pre-cloud proof. It requests neither secret,
makes no network call, and writes neither dataset nor live operator state. It
validates the sealed/readiness binding, classifies current and archived operator
receipts, verifies the latest sanitized empty-inventory budget snapshot, then
copies only `_operator` to a temporary directory and simulates terminal archive,
hash-chain index reconciliation, stale-lock recovery, attempt creation, and
atomic current-receipt creation. Proceed only when
`ready_for_live_inventory=true`, `ready_for_create_after_live_gates=true`,
`local_blockers=[]`, both active/unclean counts are zero, and API/mutation counts
are zero. Live authenticated inventory and billing still run again after the
RunPod key is supplied and before any create.
Because reconciliation is persisted immediately before the attempt receipt,
the next plan accepts exactly one additional receipt only when removing that
receipt reproduces the snapshot's recorded source count and SHA-256; replacement
or multiple unbound additions fail closed.

`ReconcileLocalReceipts` is local-only and must be used when CloudPlan reports
an archive-index or partial-file reconciliation blocker. It touches only
`<CloudRuntimeRoot>\_operator`, asks for neither Mapillary nor RunPod credentials,
and makes no network call. A terminal `cleanup_verified` current receipt is moved
losslessly into the immutable archive, valid interrupted receipt temporaries are
recovered, invalid temporaries are content-addressed into quarantine, and the
hash-chain index is rebuilt atomically. Repeating the action against the same
state is a byte-level no-op. Active or cleanup-unverified current receipts remain
typed blockers and are never archived by this action.

The dataset `run_id` is stable across retries; every cloud lifecycle receives a
new 32-hex `attempt_id`. A new terminal receipt uses the exact canonical grammar
`v2--<run_id>--<attempt_id>--<failed|terminated>--<16 lowercase SHA hex>.json`.
The formatter and parser round-trip byte-exactly, reject alternate separators,
Unicode and case drift, and never overwrite a content-prefix collision. The
atomic archive index is deterministic and hash-chained. Existing `<run_id>.json`
and historical unversioned compound receipts remain byte-for-byte unchanged and
are classified as legacy by the same parser. Byte-identical archive retries
succeed without another copy; semantic legacy duplicates remain one duplicate
group. Budget evidence and
the create idempotency key use attempt identity. An active/stale receipt or a
terminal receipt without verified cleanup blocks with a typed error.
If SSH-key or transfer preparation fails after current-receipt creation but
before create is entered, the attempt is safely terminal and cleanup-verified.
After create entry, only receipt-bound termination plus restored inventory can
set cleanup verification.
An active-stage receipt is reported as running only while its supervisor PID is
live. A dead PID fails before the RunPod prompt with
`STALE_OPERATOR_RECEIPT_REQUIRES_TERMINATE`.
Operator blockers are classified before the budget snapshot is required, so a
missing snapshot cannot mask an active, stale, or unclean receipt.

When acquisition is required, `Execute`/`Resume` read the Mapillary token and
RunPod key separately with `Read-Host -AsSecureString`. The Mapillary token
exists only in the local acquisition child environment and is cleared before the
RunPod key is requested. The RunPod key is requested only after the read-only
dataset integrity gate and exists only for lifecycle children. Neither value is
passed on argv or written to state, logs, archives or receipts; both process
variables are cleared in `finally` blocks and the secure BSTR is zeroed.

The fine-tuning path has a hard USD 3 ceiling per run (`2.95` soft stop,
`2.99` termination stop) and a fail-closed USD 10 historical ceiling. Before
any mutation, an authenticated read-only client inventories Pods, endpoints,
network volumes, and templates, then reads the account billing snapshot. Any
inventory object or unrecognized active hourly spend blocks creation. The
supervisor reconciles provider-billed totals with de-duplicated local receipts
and a conservative timestamp-by-hourly-price plus disk estimate. A terminated,
cleanup-verified Pod releases only the unused part of its reservation; active or
cleanup-unverified runs retain their full exposure. The sanitized reconciliation
receipt stores totals and hashes, never the provider body or credentials. A new
run is refused with `BUDGET_INSUFFICIENT` unless billed cost, unbilled estimate,
active exposure, and the full USD 3 proposal fit under USD 10;
`HISTORICAL_BUDGET_ALREADY_EXCEEDED` is reserved for verified actual spend over
the cap.

The v4 acquisition supervisor checkpoint is written atomically beside the
existing page and media checkpoints. Migration is additive: it records the v3
metadata-checkpoint SHA and preserves its run ID, first-seen rows and cursor
state. It contains the pending cell queue, completed/partial/failed cells,
active city/bbox/depth/cursor, first-seen ledger, city/global quotas, typed
failure ledger, request/wall/media budgets, last atomic transition and coverage
summary. Server, timeout and transport exhaustion subdivide the active cell and
continue bounded work; paging loops/malformed paging quarantine the cell before
continuing. Eight consecutive failures without a successful page pause the run.
Media authorization failures are terminal, rate limiting pauses globally, and
unavailable/retry-exhausted individual media are checkpointed and not retried
forever.

Before any Pod create, the sealed corpus must pass multi-region/city minimums,
checksum inventory, provenance, image-label binding, duplicate and
sequence/contributor concentration gates, plus secret/private-path scans. A
failure produces one `DATASET_NOT_READY_FOR_TRAINING` report with
`gpu_started=false` and `cloud_mutations=0`.

Metadata split feasibility is evaluated before any image download. Contributor
and sequence connected components are assigned to one role, the one-kilometre
reference/locked-holdout exclusion is enforced, and the locked per-city
minimums are never lowered. A feasible plan downloads only 830 primary records
plus at most ten reserves for each of the twenty city/role buckets (1,030
candidates maximum), not the whole metadata corpus. Decode failures, exact
duplicates, and perceptual duplicates within Hamming distance four are removed
after download; deterministic reserves refill the affected bucket before the
split can seal.

`training-readiness.json` distinguishes metadata feasibility from post-media
readiness and reports sanitized required/available/deficit values. A genuinely
infeasible checkpoint enters `DATASET_SUPPLEMENTAL_REQUIRED` with explicit
request (512), metadata-record (600), media-byte (zero), and wall-time
(30-minute) supplemental caps. Only deficient buckets are targeted, completed
cells are not queried again, existing metadata/media remain intact, provider
failures keep the v4 quarantine behavior, and a later Resume rebuilds the split
from the preserved checkpoint. Repeating Resume without new admissible data
reproduces the same bounded report instead of another generic failure loop.

Training freezes most of pinned MegaLoc and fine-tunes a bounded tail with a
batch-hard cosine metric objective, AdamW, real backward/optimizer steps,
mixed precision, deterministic seed and train-only augmentation. The effective
batch remains 16: four-example CUDA microbatches use deterministic activation
replay and one full-batch loss, then optimizer/scheduler step only at the complete
accumulation boundary. Checkpoints are forbidden between those boundaries. CUDA
OOM halves the current microbatch atomically; batch-one failure is terminal and
stage typed. Baseline, validation and holdout inference use eval/inference mode,
four-example bounded batches and the same backoff without changing asset order.
Epoch checkpoints support interruption resume; validation early stopping and the
345-minute Pod wall limit remain hard bounds. The locked holdout is described
once, only after the fine-tuned validation threshold is locked. Outputs include
pretrained/fine-tuned validation comparisons, final holdout benchmark, changed
weight hashes and provenance. Regression never changes the production model
automatically.

The 345-minute parameter is the total receipt-bound attempt ceiling, not a child
training deadline. One authoritative plan reserves 3,600 seconds for allocation,
SSH and transfer, 600 seconds for bootstrap, and 300 seconds for failure salvage;
the training child is capped at 16,200 seconds (270 minutes). Immediately after
bootstrap, the supervisor deducts actual elapsed time using the monotonic clock,
retains the salvage reserve, and only then converts the remaining bounded duration
to one future epoch value at the process boundary. Training converts that epoch
once and uses monotonic time internally. Supported total values are 105 through
345 minutes; the recommended full-training value is 345. Invalid values block
before a RunPod key/client or create attempt. If less than 30 minutes remains for
training, the existing Pod fails with
`REMOTE_TRAINING_TIME_REMAINING_INSUFFICIENT` without launching the child.

`TrainingDeadlinePlan` is the required read-only check before any later paid
action. It reports every reserve, the resulting child budget, supported bounds,
typed blockers and zero RunPod/API/cloud-mutation counters. A child-side invalid
epoch is preserved as `REMOTE_TRAINING_DEADLINE_INVALID`, including the sanitized
`TrainingError`, message code, process exit code and traceback tail.

`TrainingMemoryPlan` is the second required read-only gate. It accepts only the
real local CUDA production-shape receipt bound to the current readiness,
sealed-assets, checksum-inventory and model hashes. The receipt covers all 450
reference descriptors, all 150 calibration descriptors, one effective training
batch, checkpoint/new-process resume and a calibration-backed holdout-path probe
with locked-holdout access zero. It reports measured allocated/reserved peaks,
headroom and provider-memory minimum with zero network/API/create/cloud/dataset
write counters. The direct supervisor repeats this check before its first RunPod
inventory call. Offers are filtered by provider-reported VRAM and the $0.50/hour
cap; a measured requirement above 24 GiB admits only the 48 GiB A40/A6000 class.
No eligible offer fails as `GPU_MEMORY_PLAN_UNSATISFIED` before create.

Remote bootstrap treats image allocation and training compatibility as separate
gates. The create payload retains the exact digest-pinned `imageName`; when create
or authenticated GET returns that field, it must byte-match the request or the Pod
is cleaned up with `REMOTE_IMAGE_IDENTITY_MISMATCH`. The official supported minor
pairs are Torch/torchvision 2.7/0.22, 2.8/0.23 and 2.9/0.24, but a listed pair is
not sufficient by itself. Bootstrap records the full Python, Torch, torchvision,
CUDA, compiled-ops and small CUDA-operation report. For the pinned CPython 3.12,
CUDA 12.8 image, missing torchvision is repaired only from the pre-transferred
`torchvision-0.24.1+cu128-cp312-cp312-manylinux_2_28_x86_64.whl` whose SHA-256 is
`cf84eae1d2d12a7d261a7496eca00dd927b71792011b1e84d4162c950eb3201d`.
The wheel comes from the official PyTorch CUDA 12.8 wheel index and is installed
with `--no-index --no-deps`; Pod-side package-index resolution and Torch download
are prohibited. The remaining lightweight lock is also transferred and installed
offline with exact hashes. Pre-cloud validation fails before RunPod credentials or
mutation with `REMOTE_TORCHVISION_COMPANION_MISSING` or
`REMOTE_TORCHVISION_WHEEL_HASH_MISMATCH` when that inventory is incomplete.

Bootstrap hashes the imported Torch module path and file and records its version,
CUDA version, CPython ABI, device and inode before installing the companion. The
project preflight must reproduce that identity exactly or fail with
`REMOTE_TORCH_IDENTITY_CHANGED`. A companion install failure is preserved as
`REMOTE_TORCHVISION_INSTALL_FAILED`; no generic dependency wrapper may replace
these typed codes.

Before `PHASE3F_CLOUD_JOB_STARTED`, a separate 180-second smoke verifies vendor
and model hashes, imports and loads MegaLoc, and performs a synthetic CUDA AMP
forward, finite loss, backward and optimizer step. It does not open the locked
holdout, write a checkpoint or advance production state. Only
`PHASE3F_REMOTE_TRAINING_SMOKE_PASSED` permits training. Failures salvage the base
report, dependency report, environment identity, smoke report and redacted logs.

## Preferred local-first execution

Phase 3F acquisition and GPU compute are separate checkpoints. Acquisition runs
on Windows with CPU/network only, persists every completed Mapillary page and
asset checkpoint atomically, and seals `sealed-acquisition` with a complete
checksum inventory. It cannot load MegaLoc or CUDA. Compute refuses to start
until that sealed inventory, selection lock, split lock, and every media hash
verify. It then removes `MAPILLARY_ACCESS_TOKEN`, enforces the Hugging Face and
Transformers offline modes, blocks INET socket connections, and uses only the
existing local model and vendor artifacts.

The token must already be exported in the PowerShell process environment. It is
never accepted on argv or written to state. Run each command from any working
directory:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\geoSearch\scripts\phase3f-local.ps1" -Action AcquireOnly
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\geoSearch\scripts\phase3f-local.ps1" -Action DiagnoseAcquisition
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\geoSearch\scripts\phase3f-local.ps1" -Action Status
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\geoSearch\scripts\phase3f-local.ps1" -Action Resume
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\geoSearch\scripts\phase3f-local.ps1" -Action ComputeOnly
```

The runtime defaults to `D:\AtlasLensRuntime\phase3f-local`; C-drive runtime is
refused and child `TEMP`/`TMP` are redirected below that D-drive root. A fresh
acquisition also requires at least 8 GiB free on that drive and admits at most
7 GiB of downloaded media inside an 8 GiB acquisition runtime cap. HTTP
400/401/403 remains terminal; 429, 5xx, timeout, and transport failures use the
existing bounded retry policy. No model or dataset download fallback exists.

Cleanup is preview-first. Read the 32-hex `run_id` from `Status`, preview, then
repeat with its exact value:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\geoSearch\scripts\phase3f-local.ps1" -Action Cleanup
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\geoSearch\scripts\phase3f-local.ps1" -Action Cleanup -Execute -ConfirmRunId "<RUN_ID_FROM_STATUS>"
```

The sealed acquisition is at
`D:\AtlasLensRuntime\phase3f-local\<run-id>\sealed-acquisition`; offline
descriptor, index, benchmark, receipt, and checksum results are at
`D:\AtlasLensRuntime\phase3f-local\<run-id>\compute-output`.

`DiagnoseAcquisition` is available only for a resumable
`MAPILLARY_API_SERVER_RETRY_EXHAUSTED` run. It performs exactly ten
metadata-only `/images` requests with retry zero, retains no response body or
full URL, hashes allowlisted request IDs, and never downloads imagery. The
2026-07-20 diagnostic isolated bbox size: exact, repeated, minimal-field and
reduced-limit requests returned 500 while baseline and two quarter cells
returned 200. Phase 3F therefore quarters every former cell in stable
SW/SE/NW/NE order; fields, limit, token, and retry policy are unchanged.
The page engine checkpoint v3 binds both the original and adaptive ordered cell
plans by hash; the additive v4 supervisor checkpoint described above seals its
state and budgets without rewriting accepted rows.
If a cell's first request or a later cursor exhausts server retries, only that
cell is atomically replaced in-place by SW/SE/NW/NE children for the next
Resume. Accepted metadata rows remain first-seen deduplicated and unchanged.
Subdivision is bounded at depth four, a minimum child area of 0.000001 square
degrees, and 4,096 cells per city; crossing a bound is a typed terminal error.
An old failed v2 checkpoint is migrated and subdivided in one atomic replace;
only a completely empty v1 checkpoint can be migrated, while progressed v1 is
refused. The named `ce23d58c42bf76e6de0b0117725e3ea7` run is already v3 at
city 0/cell 8 with its eight pages, 469 rows, and 8/16/0 page/request/rejected
counters preserved. Resume continues that same run ID at the first depth-one
child.

Metadata capacity is a deterministic balance policy, not permission to seal a
partial dense city. Each of the sixteen cities admits at most 600 first-seen
rows, and the global metadata quota is 9,600. When a page crosses a city quota,
only the ordered prefix needed to reach 600 is atomically checkpointed; cursor
state for that city is cleared and acquisition advances to the next city. The
underlying generic client retains its 20,000-item safety ceiling. Unexpected
generic item-cap or premature global-quota exhaustion is terminal rather than a
Resume loop. For the current run, İstanbul has 569 unique rows, cells 0-8 are
complete and cell 9 is active; all other cities remain unaudited. No imagery has
been downloaded. The next Resume may admit at most 9,031 further metadata rows
before the global quota, and sealing remains unavailable until all sixteen city
audits and multi-region selection gates pass.

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
account inventory shows one unique receipt-bound Pod and no unexpected Pod,
independent endpoint, independent network volume or independent template. Pod
JSON fields such as `endpointId`, `networkVolume`, `networkVolumeId`,
`templateId`, machine data, ephemeral/container volume, ports and public IP are
attributes or associations of that Pod and are not counted as additional account
resources. This evidence is recorded as
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
is normalized from provider responses at top-level `gpuCount`, `gpu.count`, or
`machine.gpuType.count`. The create request's `gpuCount=1` is not provider
evidence. Only JSON integer `1` is accepted: booleans, strings and floats are
`POD_GPU_COUNT_INVALID`; integer values other than one or conflicting recognized
paths are `POD_GPU_COUNT_MISMATCH`. All simultaneously present non-null paths
must agree. Display names and `machine.minPodGpuCount` are not allocation
evidence. Missing/null count paths remain pending. Allocation
success requires the exact bound ID and run name, `RUNNING`, count one, selected
GPU ID, valid on-demand price/rental evidence, no proof of `interruptible=true`,
and account inventory containing only the bound Pod, with zero unexpected Pods,
independent endpoints, independent network volumes and independent templates.
Duplicate list rows for the same bound Pod are counted once. If exact bound-Pod
GET succeeds while the Pod list has not converged, allocation remains pending
within the same deadline and never creates another Pod. Public IP and port
mapping are eventual connectivity fields and are not allocation invariants.
A mismatch terminates immediately. If every count path remains missing/null,
expiration raises `POD_GPU_COUNT_ATTESTATION_TIMEOUT`; other incomplete
allocation evidence retains `POD_GPU_ATTESTATION_TIMEOUT`. Successful allocation
records `gpu_attestation_outcome=passed`, the normalized GPU-ID path, normalized
count path/value, then ends allocation polling before connectivity begins.
Receipt v2 stores
only allowlisted fields or a hash for mismatching GPU text, plus poll count/time,
outcome, `receipt_bound_pod_count`, `unexpected_pod_count`, `endpoint_count`,
`network_volume_count`, `template_count`, and `receipt_bound_match`. It never
logs Pod/resource IDs and remains able to read receipts written before these
fields.

After `allocation_attested_at` is recorded, connectivity uses authenticated
`GET /pods/{receipt-bound-id}` only; it never issues another create/POST. The
default monotonic deadline is 180 seconds. Polling is every two seconds for the
first 30 seconds and every five seconds thereafter. Missing, null, empty or
whitespace `publicIp`, a missing `22/tcp` exposed port mapping, or an SSH probe
that is not ready remains pending on the same Pod. Each poll rechecks exact GPU
ID and any supplied count; an already attested provider count remains valid when
a later sparse GET omits all count paths. Invalid or conflicting count evidence
retains `POD_GPU_COUNT_INVALID` or `POD_GPU_COUNT_MISMATCH`, not a connectivity
code. Terminal status, cloud type, on-demand price and absence of boolean
`interruptible=true`. Once a syntactically valid IP and port mapping exist, a
bounded batch-mode SSH `true` probe must pass before transfer starts. Probe output,
host keys, IP, port, Pod ID and credentials are never printed. The private operator
receipt retains its pre-existing exact Pod ID solely as the termination binding;
new connectivity telemetry contains only `public_ip_present`, `tcp_port_present`,
poll count, elapsed time, `ssh_ready`, outcome and a typed failure code. Missing IP/port at the deadline is
`POD_CONNECTIVITY_TIMEOUT`; a reached endpoint whose SSH probe never succeeds is
`POD_SSH_READINESS_TIMEOUT`; a vanished Pod is
`POD_CONNECTIVITY_POD_MISSING`. GPU/count, interruptible, cloud or price drift and
`FAILED`/`EXITED`/`TERMINATED` status fail immediately and enter the same
receipt-bound cleanup path.
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
`POD_GPU_ATTESTATION_TIMEOUT`, `POD_GPU_COUNT_ATTESTATION_TIMEOUT`,
`POD_GPU_COUNT_INVALID`, `POD_GPU_COUNT_MISMATCH`,
`POD_GPU_ATTESTATION_POD_MISSING`,
`POD_CONNECTIVITY_TIMEOUT`, `POD_SSH_READINESS_TIMEOUT`,
`POD_CONNECTIVITY_POD_MISSING`,
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
