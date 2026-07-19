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
