from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import pytest

ROOT = Path(__file__).resolve().parents[3]
START_PATH = ROOT / "scripts" / "start-investor-demo.ps1"
STOP_PATH = ROOT / "scripts" / "stop-investor-demo.ps1"
RUNTIME_PATH = ROOT / "scripts" / "runtime" / "AtlasLensRuntime.ps1"
E2E_RUNNER_PATH = ROOT / "scripts" / "run-investor-demo-e2e.ps1"
E2E_SPEC_PATH = ROOT / "apps" / "web" / "e2e" / "investor-demo-external.spec.ts"
START = START_PATH.read_text(encoding="utf-8")
STOP = STOP_PATH.read_text(encoding="utf-8")
RUNTIME = RUNTIME_PATH.read_text(encoding="utf-8")
E2E_RUNNER = E2E_RUNNER_PATH.read_text(encoding="utf-8")
E2E_SPEC = E2E_SPEC_PATH.read_text(encoding="utf-8")


def test_start_is_relocatable_and_runs_disk_gates_before_mutation() -> None:
    assert 'Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")' in START
    disk_gate = START.index("Assert-AtlasLensDiskBudget")
    first_runtime_write = START.index("New-Item -ItemType Directory")
    first_process_start = START.index("& $powershell", disk_gate)
    assert disk_gate < first_runtime_write
    assert disk_gate < first_process_start
    assert '[pscustomobject]@{ Drive = "C"; MinimumGiB = 35.0 }' in RUNTIME
    assert '[pscustomobject]@{ Drive = "D"; MinimumGiB = 8.0 }' in RUNTIME
    assert "$freeGiB -lt $requirement.MinimumGiB" in RUNTIME


def test_start_uses_only_the_required_local_worker_and_fails_closed() -> None:
    assert '"start-phase6c-workers.ps1"' in START
    assert "start-phase6b-workers.ps1" not in START
    assert "-SkipInferenceVerification" not in START
    assert 'Port = $workerPort; Label = "MegaLoc worker"' in START
    assert 'Port = $ApiPort; Label = "API"' in START
    assert 'Port = $WebPort; Label = "Web"' in START
    assert "Assert-PortFree $portGate.Port $portGate.Label" in START
    assert "Get-NetTCPConnection -State Listen -ErrorAction Stop" in START
    assert "Where-Object { [int]$_.LocalPort -eq $Port }" in START
    assert "-LocalPort $Port -ErrorAction SilentlyContinue" not in START
    assert "Get-TrustedWorkerRecord" in START
    assert "MegaLoc worker process ownership could not be verified" in START


def test_start_prepares_the_canonical_case_before_api_startup() -> None:
    worker_ready = START.index("$workerRecord = Get-TrustedWorkerRecord")
    prepare = START.index("-m atlaslens_api.cli private-demo prepare")
    receipt = START.index("$prepareReceipt.case_id", prepare)
    state_write = START.index('Write-DemoState "starting"', receipt)
    api_start = START.index("$apiProcess = Start-Process")
    assert worker_ready < prepare < receipt < state_write < api_start
    for gate in (
        "--database-url",
        "--expected-publication-sha256",
        "--expected-source-policy-sha256",
        "--expected-selection-lock-sha256",
        "--pilot-root",
        "--worker-host 127.0.0.1",
        "--worker-port",
    ):
        assert gate in START
    assert 'DATABASE_URL = $databaseUrl' in START
    assert 'PHASE6C_ENABLED = "false"' in START
    assert "private-demo CLI just created this disposable DB" in START
    assert '"-IgnoreProjectEnv", "-SkipMigration"' in START
    assert "function ConvertTo-CanonicalDemoCaseId" in START
    assert '[Guid]::TryParseExact($text, "D", [ref]$parsed)' in START
    assert "$text -notmatch $canonicalUuidPattern" in START
    assert "[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}" in START
    assert "case_id = $demoCaseId" in START
    assert "$prepareOutput = @(& $runtime.ApiPython" in START
    assert "$prepareExitCode = $LASTEXITCODE" in START
    assert "$prepareLines.Count -ne 1" in START
    for receipt_gate in (
        'status -notin @("created", "existing", "repaired")',
        "audit_integrity_valid -ne $true",
        'analysis_mode -ne "local_only"',
        "query_bytes_retained -ne $false",
    ):
        assert receipt_gate in START


def test_start_disables_downloads_cloud_and_project_env_loading() -> None:
    for assignment in (
        'HF_DATASETS_OFFLINE = "1"',
        'HF_HUB_OFFLINE = "1"',
        'TRANSFORMERS_OFFLINE = "1"',
        'UV_OFFLINE = "1"',
        'npm_config_offline = "true"',
        'NVIDIA_VISION_ENABLED = "false"',
        'OPENAI_GEO_ENABLED = "false"',
        'MAPILLARY_ENABLED = "false"',
        'VITE_MAP_PROVIDER = "maplibre"',
        "VITE_MAP_TILE_URL = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'",
    ):
        assert assignment in START
    assert "function Remove-InvestorDemoProcessVariable" in START
    for secret_name in ("NVIDIA_API_KEY", "OPENAI_API_KEY", "MAPILLARY_ACCESS_TOKEN"):
        assert secret_name in START
    assert '"VITE_MAP_STYLE_URL"' in START
    assert "Remove-InvestorDemoProcessVariable -Snapshot $Snapshot -Name $name" in START
    assert '"-IgnoreProjectEnv"' in START
    assert "bootstrap" not in START.lower()
    assert "setup-models" not in START.lower()
    assert "pip install" not in START.lower()
    assert "npm.cmd install" not in START.lower()


def test_launcher_network_surface_is_loopback_plus_viewport_tiles_and_has_no_secret_value() -> None:
    urls = re.findall(r'https?://[^"\s]+', START + STOP)
    assert urls
    assert {urlparse(url).hostname for url in urls} <= {
        "127.0.0.1",
        "localhost",
        "tile.openstreetmap.org",
    }
    external_urls = [
        url
        for url in urls
        if urlparse(url).hostname not in {"127.0.0.1", "localhost"}
    ]
    assert external_urls == ["https://tile.openstreetmap.org/{z}/{x}/{y}.png'"]
    assert "function Remove-InvestorDemoProcessVariable" in START
    assert not re.findall(r'(?:API_KEY|ACCESS_TOKEN)\s*=\s*"[^"]+"', START)
    for forbidden in (
        "start-bitstransfer",
        "curl.exe",
        "wget.exe",
        "git clone",
        "huggingface-cli",
        "hf download",
        "winget install",
        "choco install",
    ):
        assert forbidden not in (START + STOP).lower()


def test_success_requires_case_workspace_readiness_and_prints_exact_case_url() -> None:
    health = START.index("/api/v1/health")
    ready = START.index("/api/v1/ready")
    demo_status = START.index("/api/v1/mapillary-demo/status", ready)
    api_case = START.index("/api/v1/cases/$demoCaseId", demo_status)
    web_200 = START.index("Wait-WebHttp200 -Url")
    proxy_status = START.index("/api/v1/mapillary-demo/status", demo_status + 1)
    web_case = START.index("/api/v1/cases/$demoCaseId", api_case + 1)
    workspace_pages = START.index('@("media", "evidence", "hypotheses", "audit-events")')
    audit_integrity = START.index("/audit-integrity", workspace_pages)
    ready_state = START.index('Write-DemoState "ready"', audit_integrity)
    deep_link = START.index("/?demo=investor&lang=tr&caseId=")
    success = START.index("AtlasLens investor demo is ready")
    assert (
        health
        < ready
        < demo_status
        < api_case
        < web_200
        < proxy_status
        < web_case
        < workspace_pages
        < audit_integrity
        < ready_state
        < deep_link
        < success
    )
    assert '$body.state -eq "active" -and $body.available -eq $true' in START
    assert "[int]$response.StatusCode -eq 200" in START
    assert '[string]$body.id -eq $demoCaseId' in START
    assert '$null -ne $body.PSObject.Properties["items"]' in START
    assert '$null -ne $body.PSObject.Properties["total"]' in START
    assert "$body.valid -eq $true" in START
    assert '[Uri]::EscapeDataString($demoCaseId)' in START
    assert 'Write-Host "Demo: $demoUrl"' in START
    assert "Start-Process http" not in START


def test_runtime_state_is_ignored_and_shutdown_requires_owned_identity() -> None:
    assert '.local\\run\\investor-demo' in START
    assert 'schema_version = "atlaslens-investor-demo-process-v1"' in START
    assert "process_started_at_utc" in START
    assert "launcher_started_at_utc" in START
    assert "listener_started_at_utc" in START
    assert ".local/" in (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "Validate every live target before the first termination action" in STOP
    prevalidate = STOP.index("Assert-ServiceOwnership $state.web")
    first_stop = STOP.index("Stop-OwnedService $state.web")
    assert prevalidate < first_stop
    assert "Test-RecordedStart" in STOP
    assert "Test-ProcessDescendsFrom" not in STOP
    assert "ParentProcessId -in $parents" not in STOP
    assert "Get-OwnedProcessSnapshot" in STOP
    assert "CIM binding prevents a recycled root" in STOP
    assert "Assert-OwnedRootSnapshotsCurrent" in STOP
    assert 'Role = "listener"' in STOP
    assert 'Role = "launcher"' in STOP
    assert '"vite" "node"' in STOP
    assert '"atlaslens_api.main:app" "python"' in STOP
    assert '"python" "MegaLoc worker"' in STOP
    assert "Assert-OptionalCommandMetadata" in STOP
    assert "has nonblank mismatching executable metadata" in STOP
    assert "has nonblank mismatching command-line metadata" in STOP
    assert 'Stop-Process -InputObject $process' in STOP
    assert "remained alive; trusted state was retained" in STOP
    assert "was recycled after its owned process exited; the new process was not touched" in STOP
    assert "not owned by this investor demo" in STOP
    assert "metadata.worker_path" in STOP
    assert "Assert-WorkerMetadataPathSafe" in STOP
    assert "stop-phase6c-workers.ps1" not in STOP
    assert "existing listeners were not touched" in STOP
    assert "Get-Process | Stop-Process" not in STOP
    assert 'Get-Command "taskkill.exe"' in STOP
    assert "& $taskkill /PID $processId /F" in STOP
    assert "taskkill fallback" in STOP
    assert "/T" not in STOP
    assert "Stop-Process -Name" not in STOP


def test_shutdown_gates_cleanup_and_taskkill_behind_exact_snapshot_revalidation() -> None:
    no_state_branch = STOP.index("if (-not (Test-Path -LiteralPath $statePath -PathType Leaf))")
    tree_gate = STOP.index("Assert-InvestorDemoRuntimeTreeSafe", no_state_branch)
    state_file_gate = STOP.index("Assert-InvestorDemoStateFileSafe", tree_gate)
    state_read = STOP.index("Get-Content -LiteralPath $statePath -Raw", state_file_gate)
    first_stop = STOP.index("Stop-OwnedService $state.web", state_read)
    assert tree_gate < state_file_gate < state_read < first_stop
    assert "Investor-demo state is not a regular non-reparse file" in STOP
    assert "Get-NetTCPConnection -State Listen -ErrorAction Stop" in STOP

    closed_ports = STOP.rindex("Assert-NoInvestorDemoListeners")
    worker_metadata_cleanup = STOP.rindex("Remove-TrustedWorkerMetadata")
    runtime_cleanup = STOP.rindex("Remove-InvestorDemoRuntime")
    assert first_stop < closed_ports < worker_metadata_cleanup < runtime_cleanup
    assert "trusted metadata and runtime state were retained" in STOP

    service_start = STOP.index("function Stop-OwnedService")
    service_end = STOP.index("function Remove-TrustedWorkerMetadata", service_start)
    service_body = STOP[service_start:service_end]
    service_snapshot = service_body.index("Get-OwnedProcessSnapshot")
    service_root_recheck = service_body.index(
        "Assert-OwnedRootSnapshotsCurrent", service_snapshot
    )
    service_first_stop = service_body.index(
        "Stop-Process -InputObject $process", service_root_recheck
    )
    assert service_snapshot < service_root_recheck < service_first_stop
    service_taskkill = service_body.index("& $taskkill /PID $processId /F")
    service_wait = service_body.rfind("if (-not $process.WaitForExit", 0, service_taskkill)
    service_recheck = service_body.rfind("Assert-ProcessMatchesSnapshot", 0, service_taskkill)
    assert 0 <= service_wait < service_recheck < service_taskkill
    service_race_guard = service_body.rfind(
        '$ErrorActionPreference = "Continue"', 0, service_taskkill
    )
    service_final_check = service_body.index(
        "remained alive; trusted state was retained", service_taskkill
    )
    assert service_recheck < service_race_guard < service_taskkill < service_final_check

    worker_start = STOP.index("function Stop-OwnedWorker")
    worker_end = STOP.index("function Remove-InvestorDemoRuntime", worker_start)
    worker_body = STOP[worker_start:worker_end]
    worker_taskkill = worker_body.index("& $taskkill /PID $processId /F")
    worker_wait = worker_body.rfind("if (-not $process.WaitForExit", 0, worker_taskkill)
    worker_recheck = worker_body.rfind("Assert-RecordedProcessIdentity", 0, worker_taskkill)
    assert 0 <= worker_wait < worker_recheck < worker_taskkill
    worker_race_guard = worker_body.rfind(
        '$ErrorActionPreference = "Continue"', 0, worker_taskkill
    )
    worker_final_check = worker_body.index(
        "Owned MegaLoc process remained alive", worker_taskkill
    )
    assert worker_recheck < worker_race_guard < worker_taskkill < worker_final_check
    assert "/T" not in STOP


def test_shutdown_uses_recorded_ports_and_dual_worker_identity_at_deletion() -> None:
    assert "function Assert-NoInvestorDemoListeners([object]$State)" in STOP
    assert "@($State.api, $State.web)" in STOP
    assert 'Get-PropertyValue $service "port"' in STOP
    assert "Assert-NoInvestorDemoListeners $state" in STOP
    assert "function Remove-TrustedWorkerMetadata([object]$Record)" in STOP
    assert "Revalidate both trusted records immediately before deleting" in STOP
    assert "Assert-WorkerOwnership $Record" in STOP
    assert "MegaLoc trusted records have different process start times" in STOP

    ownership_start = STOP.index("function Assert-WorkerOwnership")
    dual_start_gate = STOP.index("MegaLoc trusted records have different", ownership_start)
    absent_process = STOP.index("if ($null -eq $process)", dual_start_gate)
    remove_function = STOP.index("function Remove-TrustedWorkerMetadata")
    removal_recheck = STOP.index("Assert-WorkerOwnership $Record", remove_function)
    remove_item = STOP.index("Remove-Item -LiteralPath $workerMetadataPath", removal_recheck)
    assert ownership_start < dual_start_gate < absent_process
    assert remove_function < removal_recheck < remove_item

    worker_stop = STOP.index("Stop-OwnedWorker $state.worker")
    listener_gate = STOP.index("Assert-NoInvestorDemoListeners $state")
    metadata_remove = STOP.index("Remove-TrustedWorkerMetadata $state.worker")
    runtime_remove = STOP.rindex("Remove-InvestorDemoRuntime")
    assert worker_stop < listener_gate < metadata_remove < runtime_remove


def test_cleanup_is_contained_and_covers_stop_and_failed_start() -> None:
    for script in (START, STOP):
        assert "function Remove-InvestorDemoRuntime" in script
        assert "function Assert-InvestorDemoRuntimePath" in script
        assert "function Assert-InvestorDemoRuntimeTreeSafe" in script
        assert 'Join-Path $repoRoot ".local\\run"' in script
        assert '[IO.Path]::GetFileName($resolvedRunRoot)' in script
        assert "[IO.FileAttributes]::ReparsePoint" in script
        assert "[Collections.Generic.Stack[string]]::new()" in script
        assert "runtime tree contains a reparse point; refusing cleanup" in script
        assert '"investor-demo"' in script
        assert "Refusing cleanup outside the exact investor-demo runtime directory" in script
        assert "Remove-Item -LiteralPath $resolvedRunRoot -Recurse -Force" in script
    finalizer = START.rindex("finally {")
    failure_cleanup = START.index("$cleanupFailed = $false", finalizer)
    process_cleanup = START.index("Stop-StartedProcessTree", failure_cleanup)
    listener_check = START.index("Assert-NoStartedDemoListeners", process_cleanup)
    cleanup_guard = START.index("if (-not $cleanupFailed)", listener_check)
    cleanup = START.index("Remove-InvestorDemoRuntime", cleanup_guard)
    process_tree = START.index("function Stop-StartedProcessTree")
    cim_root_binding = START.index(
        "Bind the launcher snapshot to the same CIM table", process_tree
    )
    deepest_first = START.index("Sort-Object Depth -Descending", process_tree)
    final_root_recheck = START.index(
        "Revalidate the launcher immediately before any inferred descendant",
        deepest_first,
    )
    exact_stop = START.index("Stop-Process -InputObject $process", deepest_first)
    survivor_check = START.index("remained alive; runtime state was retained", exact_stop)
    assert (
        process_tree
        < cim_root_binding
        < deepest_first
        < final_root_recheck
        < exact_stop
        < survivor_check
        < failure_cleanup
    )
    assert failure_cleanup < process_cleanup < listener_check < cleanup_guard < cleanup
    assert "Get-CimInstance -ClassName Win32_Process -ErrorAction Stop" in START
    for script in (START, STOP):
        stale_parent_gate = script.index(
            "($startedAtUtc - $parentSnapshot.StartedAtUtc).TotalSeconds -lt -1.0"
        )
        next_snapshot = script.index("$snapshotById[$processId]", stale_parent_gate)
        assert "continue" in script[stale_parent_gate:next_snapshot]
        assert "an older process cannot" in script[stale_parent_gate:next_snapshot]
    assert "$workerStarted -and -not $cleanupFailed" in START
    assert "MegaLoc cleanup was skipped so its trusted metadata remains" in START
    final_owned_check = STOP.index("Owned MegaLoc process remained alive")
    stop_cleanup = STOP.rindex("Remove-InvestorDemoRuntime")
    assert final_owned_check < stop_cleanup
    assert "not recoverable" in STOP


def test_external_investor_demo_gate_uses_only_the_official_loopback_stack() -> None:
    for variable in (
        "ATLASLENS_E2E_BASE_URL",
        "ATLASLENS_INVESTOR_DEMO_E2E",
        "ATLASLENS_INVESTOR_DEMO_URL",
        "ATLASLENS_PHASE3E_SCREENSHOT_DIR",
    ):
        assert variable in E2E_SPEC
        assert variable in E2E_RUNNER
    assert 'parsed.hostname !== "127.0.0.1"' in E2E_SPEC
    assert 'context.route("**/*"' in E2E_SPEC
    assert 'route.abort("blockedbyclient")' in E2E_SPEC
    assert 'context.routeWebSocket("**/*"' in E2E_SPEC
    assert 'reason: "non-loopback blocked"' in E2E_SPEC
    assert 'url.hostname === "tile.openstreetmap.org"' in E2E_SPEC
    assert "isOsmViewportTile(requestUrl)" in E2E_SPEC
    assert 'contentType: "image/png"' in E2E_SPEC
    assert "controlledMapTile" in E2E_SPEC
    assert "failMockTiles" in E2E_SPEC
    assert ".body()" not in E2E_SPEC
    assert ".json()" not in E2E_SPEC
    for contract in (
        "investor-route-notice",
        "investor-authorized-cases",
        "investor-authorized-case-open",
        "page.reload()",
        "randomUUID()",
        "Yeniden dene",
        "main.investor-workspace",
        "© OpenStreetMap contributors",
        "Alt harita çevrimdışı",
        "{ width: 1920, height: 1080 }",
        "{ width: 390, height: 844 }",
    ):
        assert contract in E2E_SPEC

    start_call = E2E_RUNNER.index(
        "$launcherProcess = [Diagnostics.Process]::new()"
    )
    stop_call = E2E_RUNNER.index(
        "& $powershell -NoProfile -ExecutionPolicy Bypass -File $stopScript"
    )
    finalizer = E2E_RUNNER.rfind("finally {", start_call, stop_call)
    assert 0 <= start_call < finalizer < stop_call
    assert '$cmd = (Get-Command "cmd.exe" -ErrorAction Stop).Source' in E2E_RUNNER
    assert "$launcherInfo.FileName = $cmd" in E2E_RUNNER
    assert "$launcherInfo.WorkingDirectory = $outsideWorkingDirectory" in E2E_RUNNER
    assert "$launcherInfo.UseShellExecute = $false" in E2E_RUNNER
    assert '1>`"$launcherStdoutPath`"' in E2E_RUNNER
    assert '2>`"$launcherStderrPath`"' in E2E_RUNNER
    assert "$launcherProcess.WaitForExit()" in E2E_RUNNER
    assert "$startExitCode = $launcherProcess.ExitCode" in E2E_RUNNER
    assert "$launcherProcess.Dispose()" in E2E_RUNNER
    assert "$startScript 2>&1 |" not in E2E_RUNNER
    assert "Remove-Item -LiteralPath $capturePath -Force" in E2E_RUNNER
    assert "Get-OfficialDemoUrl $startupOutput" in E2E_RUNNER
    assert (
        '$outsideWorkingDirectory = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())'
        in E2E_RUNNER
    )
    assert '"atlaslens-phase3e-visual-"' in E2E_RUNNER
    assert 'Write-Host "Visuals: $visualRoot"' in E2E_RUNNER
    assert "Push-Location -LiteralPath $outsideWorkingDirectory" in E2E_RUNNER
    assert "$externalSpec = \"e2e/investor-demo-external.spec.ts\"" in E2E_RUNNER
    assert "run test:e2e -- --project=chromium $externalSpec --reporter=list" in E2E_RUNNER
    assert "Assert-InvestorDemoCleanup" in E2E_RUNNER
    assert "Get-NetTCPConnection -State Listen -ErrorAction Stop" in E2E_RUNNER
    assert "Test-Path -LiteralPath $runtimeRoot" in E2E_RUNNER
    assert "Test-Path -LiteralPath $workerMetadata" in E2E_RUNNER


@pytest.mark.skipif(shutil.which("powershell.exe") is None, reason="Windows PowerShell unavailable")
def test_start_case_id_helper_enforces_the_frontend_rfc_uuid_contract() -> None:
    path = str(START_PATH).replace("'", "''")
    command = f"""
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    '{path}', [ref]$tokens, [ref]$errors
)
if ($errors.Count) {{ throw ($errors | ForEach-Object Message | Out-String) }}
$function = $ast.FindAll({{
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq 'ConvertTo-CanonicalDemoCaseId'
}}, $true) | Select-Object -First 1
Invoke-Expression $function.Extent.Text
$canonicalUuidPattern = `
    '^[0-9a-f]{{8}}-[0-9a-f]{{4}}-[1-8][0-9a-f]{{3}}-[89ab][0-9a-f]{{3}}-[0-9a-f]{{12}}$'
function Assert-Rejected([string]$Value) {{
    $rejected = $false
    try {{ ConvertTo-CanonicalDemoCaseId $Value | Out-Null }}
    catch {{ $rejected = $true }}
    if (-not $rejected) {{ throw "Accepted invalid UUID: $Value" }}
}}
$valid = ConvertTo-CanonicalDemoCaseId '123E4567-E89B-52D3-A456-426614174000'
if ($valid -ne '123e4567-e89b-52d3-a456-426614174000') {{ throw 'UUID was not canonicalized.' }}
Assert-Rejected '00000000-0000-0000-0000-000000000000'
Assert-Rejected '123e4567-e89b-92d3-a456-426614174000'
Assert-Rejected '123e4567-e89b-52d3-7456-426614174000'
"""
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def _run_stop_helper_probe(body: str) -> subprocess.CompletedProcess[str]:
    helper_names = (
        "Get-PropertyValue",
        "Test-RecordedStart",
        "Assert-RecordedProcessIdentity",
        "Assert-OptionalCommandMetadata",
        "Assert-ExpectedPortOwner",
        "Assert-ListenerStartRelationship",
    )
    names = ",".join(f'"{name}"' for name in helper_names)
    path = str(STOP_PATH).replace("'", "''")
    command = f"""
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    '{path}', [ref]$tokens, [ref]$errors
)
if ($errors.Count) {{ throw ($errors | ForEach-Object Message | Out-String) }}
foreach ($name in @({names})) {{
    $function = $ast.FindAll({{
        param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
            $node.Name -eq $name
    }}, $true) | Select-Object -First 1
    if ($null -eq $function) {{ throw "Missing helper $name" }}
    Invoke-Expression $function.Extent.Text
}}
$listenerStartMaximumDelaySeconds = 30.0
$ErrorActionPreference = 'Stop'
try {{
{body}
}}
catch {{
    Write-Error $_.Exception.Message
    exit 1
}}
"""
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )


@pytest.mark.skipif(shutil.which("powershell.exe") is None, reason="Windows PowerShell unavailable")
def test_stop_identity_helpers_accept_only_unavailable_command_metadata_fallback() -> None:
    completed = _run_stop_helper_probe(
        r"""
    $started = (Get-Date).AddSeconds(-2)
    $recorded = $started.ToUniversalTime().ToString('O')
    $launcher = [pscustomobject]@{ Id = 4242; StartTime = $started; ProcessName = 'powershell' }
    Assert-RecordedProcessIdentity $launcher 4242 $recorded 'powershell' 'web launcher'

    $nullFallback = Assert-OptionalCommandMetadata `
        $null 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' `
        'D:\geoSearch\scripts\start-web.ps1' 'powershell' 'web launcher'
    if (-not $nullFallback) { throw 'Null metadata did not select fallback.' }

    $blank = [pscustomobject]@{ ExecutablePath = ''; CommandLine = '' }
    $blankFallback = Assert-OptionalCommandMetadata `
        $blank '' 'vite' 'node' 'web listener'
    if (-not $blankFallback) { throw 'Blank metadata did not select fallback.' }

    $matching = [pscustomobject]@{
        ExecutablePath = 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
        CommandLine = 'powershell.exe -File D:\geoSearch\scripts\start-web.ps1'
    }
    $matchingFallback = Assert-OptionalCommandMetadata `
        $matching $matching.ExecutablePath 'D:\geoSearch\scripts\start-web.ps1' `
        'powershell' 'web launcher'
    if ($matchingFallback) { throw 'Complete matching metadata incorrectly selected fallback.' }

    $listener = [pscustomobject]@{ Id = 5151; StartTime = $started; ProcessName = 'node' }
    Assert-RecordedProcessIdentity $listener 5151 $recorded 'node' 'web listener'
    $worker = [pscustomobject]@{ Id = 6161; StartTime = $started; ProcessName = 'python' }
    Assert-RecordedProcessIdentity $worker 6161 $recorded 'python' 'MegaLoc worker'
    Assert-ExpectedPortOwner 5151 5151 'web listener'
    Assert-ListenerStartRelationship $started.ToUniversalTime() `
        $started.AddSeconds(1).ToUniversalTime() 'web'
    Write-Output 'fallback-ok'
"""
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "fallback-ok" in completed.stdout


@pytest.mark.skipif(shutil.which("powershell.exe") is None, reason="Windows PowerShell unavailable")
def test_stop_identity_helpers_reject_mismatch_foreign_owner_and_recycled_pid() -> None:
    completed = _run_stop_helper_probe(
        r"""
    function Assert-Fails([scriptblock]$Action, [string]$ExpectedMessage) {
        $failed = $false
        try { & $Action }
        catch {
            $messageIndex = $_.Exception.Message.IndexOf(
                $ExpectedMessage, [StringComparison]::OrdinalIgnoreCase
            )
            if ($messageIndex -lt 0) {
                throw "Unexpected refusal: $($_.Exception.Message)"
            }
            $failed = $true
        }
        if (-not $failed) { throw "Expected refusal containing: $ExpectedMessage" }
    }

    $started = (Get-Date).AddSeconds(-2)
    $recorded = $started.ToUniversalTime().ToString('O')
    $badExecutable = [pscustomobject]@{
        ExecutablePath = 'C:\foreign\powershell.exe'
        CommandLine = 'powershell.exe -File D:\geoSearch\scripts\start-web.ps1'
    }
    Assert-Fails {
        Assert-OptionalCommandMetadata $badExecutable `
            'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' `
            'D:\geoSearch\scripts\start-web.ps1' 'powershell' 'web launcher'
    } 'nonblank mismatching executable metadata'

    $badCommand = [pscustomobject]@{
        ExecutablePath = 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
        CommandLine = 'powershell.exe -File D:\foreign\script.ps1'
    }
    Assert-Fails {
        Assert-OptionalCommandMetadata $badCommand $badCommand.ExecutablePath `
            'D:\geoSearch\scripts\start-web.ps1' 'powershell' 'web launcher'
    } 'nonblank mismatching command-line metadata'

    Assert-Fails { Assert-ExpectedPortOwner 9999 5151 'web listener' } 'instead of trusted PID'
    $recycled = [pscustomobject]@{ Id = 5151; StartTime = $started; ProcessName = 'node' }
    Assert-Fails {
        Assert-RecordedProcessIdentity $recycled 5151 `
            $started.AddSeconds(-10).ToUniversalTime().ToString('O') 'node' 'web listener'
    } 'does not match trusted state'
    $wrongName = [pscustomobject]@{ Id = 5151; StartTime = $started; ProcessName = 'python' }
    Assert-Fails {
        Assert-RecordedProcessIdentity $wrongName 5151 $recorded 'node' 'web listener'
    } 'does not match trusted state'
    Assert-Fails {
        Assert-ListenerStartRelationship $started.ToUniversalTime() `
            $started.AddSeconds(31).ToUniversalTime() 'web'
    } 'not after or near'
    Write-Output 'refusals-ok'
"""
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "refusals-ok" in completed.stdout


@pytest.mark.skipif(shutil.which("powershell.exe") is None, reason="Windows PowerShell unavailable")
@pytest.mark.parametrize("path", [START_PATH, STOP_PATH, E2E_RUNNER_PATH])
def test_investor_demo_scripts_parse(path: Path) -> None:
    command = (
        "$tokens=$null;$errors=$null;"
        f"[System.Management.Automation.Language.Parser]::ParseFile('{path}',"
        "[ref]$tokens,[ref]$errors)|Out-Null;"
        "if($errors.Count){$errors|ForEach-Object{$_.Message};exit 1}"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
