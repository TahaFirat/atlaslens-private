[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$ApiPort = 8000,
    [ValidateRange(1, 65535)]
    [int]$WebPort = 5173,
    [AllowEmptyString()][string]$RuntimeRoot = $(
        if ($env:ATLASLENS_RUNTIME_ROOT) { $env:ATLASLENS_RUNTIME_ROOT }
        else { "C:\AtlasLensRuntime" }
    ),
    [ValidateSet("cpu", "cuda")]
    [string]$MegaLocDevice = "cuda",
    [ValidateRange(15, 300)]
    [int]$StartupTimeoutSeconds = 180
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$runtimeSupport = Join-Path $PSScriptRoot "runtime\AtlasLensRuntime.ps1"
$workerStartScript = Join-Path $PSScriptRoot "start-phase6c-workers.ps1"
$workerStopScript = Join-Path $PSScriptRoot "stop-phase6c-workers.ps1"
$apiScript = Join-Path $PSScriptRoot "start-api.ps1"
$webScript = Join-Path $PSScriptRoot "start-web.ps1"
$powershell = (Get-Command "powershell.exe" -ErrorAction Stop).Source
$runRoot = Join-Path $repoRoot ".local\run\investor-demo"
$logRoot = Join-Path $runRoot "logs"
$dataRoot = Join-Path $runRoot "data"
$temporaryStorage = Join-Path $runRoot "temporary-uploads"
$statePath = Join-Path $runRoot "state.json"
$databasePath = Join-Path $dataRoot "investor-demo.sqlite"
$databaseUrl = "sqlite:///$($databasePath.Replace('\', '/'))"
$bundlePath = "C:\AtlasLensPilot\mapillary-demo\derived\mapillary-faiss-index"
$pilotRoot = "C:\AtlasLensPilot\mapillary-demo"
$publicationSha256 = "bcb538df0ed3205bc70ca0383f2c88ad88cea849fdeb82e690756b18028780a3"
$sourcePolicySha256 = "72d51363f2b63de368d34d4d7bb2fc1145f93dc0469e7026100976732dfde209"
$selectionLockSha256 = "8d8ad89d3c9045a6d5aa164854bee4b11d56bd0409ce51e539678b65c596d6f4"
$workerPort = 8794
$canonicalUuidPattern = "^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
$workerMetadataPath = Join-Path $repoRoot ".local\run\phase6c-workers\megaloc.json"
$apiLogOut = Join-Path $logRoot "api.out.log"
$apiLogErr = Join-Path $logRoot "api.err.log"
$webLogOut = Join-Path $logRoot "web.out.log"
$webLogErr = Join-Path $logRoot "web.err.log"
$apiProcess = $null
$webProcess = $null
$workerRecord = $null
$apiRecord = $null
$webRecord = $null
$demoCaseId = $null
$workerStarted = $false
$startupComplete = $false
$environmentSnapshot = @{}

. $runtimeSupport

function Assert-InvestorDemoRuntimePath {
    $expectedRunRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot ".local\run\investor-demo"))
    $resolvedRunRoot = [IO.Path]::GetFullPath($runRoot)
    if (-not [string]::Equals($resolvedRunRoot, $expectedRunRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Investor-demo runtime path is not the exact expected directory."
    }
    foreach ($candidate in @(
        (Join-Path $repoRoot ".local"),
        (Join-Path $repoRoot ".local\run"),
        $resolvedRunRoot
    )) {
        if (-not (Test-Path -LiteralPath $candidate)) { continue }
        $item = Get-Item -LiteralPath $candidate -Force
        if (
            -not $item.PSIsContainer -or
            ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
        ) {
            throw "Investor-demo runtime path contains a file or reparse point; refusing access."
        }
    }
}

function Assert-InvestorDemoRuntimeTreeSafe {
    Assert-InvestorDemoRuntimePath
    if (-not (Test-Path -LiteralPath $runRoot -PathType Container)) { return }
    $pending = [Collections.Generic.Stack[string]]::new()
    $pending.Push($runRoot)
    while ($pending.Count -gt 0) {
        $directory = $pending.Pop()
        foreach ($item in @(Get-ChildItem -LiteralPath $directory -Force)) {
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Investor-demo runtime tree contains a reparse point; refusing cleanup."
            }
            if ($item.PSIsContainer) { $pending.Push($item.FullName) }
        }
    }
}

function Quote-ProcessArgument([string]$Value) {
    if ($Value.Contains('"')) { throw "Process argument contains an unsupported quote." }
    return '"' + $Value + '"'
}

function Get-ListeningConnections([int]$Port) {
    return @(
        Get-NetTCPConnection -State Listen -ErrorAction Stop |
            Where-Object { [int]$_.LocalPort -eq $Port }
    )
}

function Assert-PortFree([int]$Port, [string]$Label) {
    $listeners = @(Get-ListeningConnections $Port)
    if ($listeners.Count -eq 0) { return }
    $owners = @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)
    $description = @(
        foreach ($processId in $owners) {
            $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
            if ($null -eq $process) { "$processId (unknown)" }
            else { "$processId ($($process.ProcessName))" }
        }
    ) -join ", "
    throw "$Label port $Port is already in use by $description; no process was touched."
}

function Get-LoopbackPortOwner([int]$Port) {
    $listeners = @(Get-ListeningConnections $Port)
    if ($listeners.Count -eq 0) { return $null }
    if (@($listeners | Where-Object { $_.LocalAddress -ne "127.0.0.1" }).Count -gt 0) {
        throw "Port $Port is not bound exclusively to 127.0.0.1."
    }
    $owners = @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)
    if ($owners.Count -ne 1) { throw "Port $Port has ambiguous listener ownership." }
    return [int]$owners[0]
}

function Get-ProcessCommand([int]$ProcessId) {
    return Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
}

function Test-ProcessDescendsFrom([int]$ProcessId, [int]$RootProcessId) {
    $seen = [Collections.Generic.HashSet[int]]::new()
    $current = $ProcessId
    while ($current -gt 0 -and $seen.Add($current)) {
        if ($current -eq $RootProcessId) { return $true }
        $command = Get-ProcessCommand $current
        if ($null -eq $command) { return $false }
        $current = [int]$command.ParentProcessId
    }
    return $false
}

function Wait-JsonEndpoint(
    [string]$Url,
    [scriptblock]$Validate,
    [System.Diagnostics.Process]$Process,
    [string]$Label
) {
    $deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
    do {
        $Process.Refresh()
        if ($Process.HasExited) {
            throw "$Label process exited before readiness with code $($Process.ExitCode)."
        }
        try {
            $body = Invoke-RestMethod -Uri $Url -Method Get -TimeoutSec 3 -Headers @{ Accept = "application/json" }
            if (& $Validate $body) { return }
        }
        catch {
            # Readiness remains fail-closed; the bounded timeout reports the stable URL.
        }
        Start-Sleep -Milliseconds 300
    } while ((Get-Date) -lt $deadline)
    throw "$Label readiness timed out at $Url."
}

function Wait-WebHttp200([string]$Url, [System.Diagnostics.Process]$Process) {
    $deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
    do {
        $Process.Refresh()
        if ($Process.HasExited) {
            throw "Web process exited before readiness with code $($Process.ExitCode)."
        }
        try {
            $response = Invoke-WebRequest -Uri $Url -Method Get -TimeoutSec 3 -UseBasicParsing
            if ([int]$response.StatusCode -eq 200) { return }
        }
        catch {
            # Startup is still in progress.
        }
        Start-Sleep -Milliseconds 300
    } while ((Get-Date) -lt $deadline)
    throw "Web readiness timed out without HTTP 200 at $Url."
}

function New-ServiceRecord(
    [string]$Name,
    [System.Diagnostics.Process]$Launcher,
    [int]$Port,
    [string]$ScriptPath
) {
    $owner = Get-LoopbackPortOwner $Port
    if ($null -eq $owner) { throw "$Name passed HTTP readiness without a loopback listener." }
    if (-not (Test-ProcessDescendsFrom $owner $Launcher.Id)) {
        throw "$Name listener PID $owner is not owned by launcher PID $($Launcher.Id)."
    }
    $listener = Get-Process -Id $owner -ErrorAction Stop
    $launcherCommand = Get-ProcessCommand $Launcher.Id
    if (
        $null -eq $launcherCommand -or
        [string]::IsNullOrWhiteSpace($launcherCommand.CommandLine) -or
        $launcherCommand.CommandLine.IndexOf($ScriptPath, [StringComparison]::OrdinalIgnoreCase) -lt 0
    ) {
        throw "$Name launcher identity could not be verified."
    }
    return [ordered]@{
        name = $Name
        port = $Port
        launcher_pid = $Launcher.Id
        launcher_started_at_utc = $Launcher.StartTime.ToUniversalTime().ToString("O")
        launcher_executable = [string]$launcherCommand.ExecutablePath
        script_path = $ScriptPath
        listener_pid = $listener.Id
        listener_started_at_utc = $listener.StartTime.ToUniversalTime().ToString("O")
    }
}

function Get-TrustedWorkerRecord {
    if (-not (Test-Path -LiteralPath $workerMetadataPath -PathType Leaf)) {
        throw "MegaLoc readiness metadata is missing."
    }
    try { $record = Get-Content -LiteralPath $workerMetadataPath -Raw | ConvertFrom-Json }
    catch { throw "MegaLoc readiness metadata is invalid." }
    if (
        $record.schema_version -ne "atlaslens-worker-process-v1" -or
        $record.provider -ne "megaloc" -or
        [int]$record.port -ne $workerPort
    ) {
        throw "MegaLoc readiness metadata identity is invalid."
    }
    $processId = [int]$record.pid
    $owner = Get-LoopbackPortOwner $workerPort
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    $command = Get-ProcessCommand $processId
    if ($null -eq $process -or $owner -ne $processId -or $null -eq $command) {
        throw "MegaLoc worker process ownership could not be verified."
    }
    $recordedStart = [DateTimeOffset]::Parse([string]$record.process_started_at_utc).UtcDateTime
    if ([Math]::Abs(($recordedStart - $process.StartTime.ToUniversalTime()).TotalSeconds) -ge 1.0) {
        throw "MegaLoc worker PID was recycled."
    }
    $workerPath = (Join-Path $repoRoot "services\model-workers\megaloc\worker.py").Replace("/", "\")
    if (
        [string]::IsNullOrWhiteSpace($command.CommandLine) -or
        $command.CommandLine.Replace("/", "\").IndexOf($workerPath, [StringComparison]::OrdinalIgnoreCase) -lt 0
    ) {
        throw "MegaLoc worker command identity could not be verified."
    }
    return [ordered]@{
        provider = "megaloc"
        port = $workerPort
        pid = $processId
        process_started_at_utc = $process.StartTime.ToUniversalTime().ToString("O")
        worker_path = $workerPath
    }
}

function Write-DemoState([string]$Status) {
    $state = [ordered]@{
        schema_version = "atlaslens-investor-demo-process-v1"
        status = $Status
        repository_root = $repoRoot
        created_by_pid = $PID
        updated_at_utc = (Get-Date).ToUniversalTime().ToString("O")
        case_id = $demoCaseId
        worker = $workerRecord
        api = $apiRecord
        web = $webRecord
    }
    $temporary = "$statePath.$PID.tmp"
    $json = $state | ConvertTo-Json -Depth 6
    [IO.File]::WriteAllText($temporary, $json, [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $statePath -Force
}

function ConvertTo-CanonicalDemoCaseId([object]$Value) {
    $text = if ($null -eq $Value) { "" } else { [string]$Value }
    $parsed = [Guid]::Empty
    if (
        [string]::IsNullOrWhiteSpace($text) -or
        $text -notmatch $canonicalUuidPattern -or
        -not [Guid]::TryParseExact($text, "D", [ref]$parsed)
    ) {
        throw "Private-demo preparation returned an invalid case identifier."
    }
    return $parsed.ToString("D").ToLowerInvariant()
}

function Remove-InvestorDemoRuntime {
    Assert-InvestorDemoRuntimeTreeSafe
    $resolvedRunRoot = [IO.Path]::GetFullPath($runRoot)
    $allowedParent = [IO.Path]::GetFullPath((Join-Path $repoRoot ".local\run")) + [IO.Path]::DirectorySeparatorChar
    if (
        -not $resolvedRunRoot.StartsWith($allowedParent, [StringComparison]::OrdinalIgnoreCase) -or
        -not [string]::Equals(
            [IO.Path]::GetFileName($resolvedRunRoot),
            "investor-demo",
            [StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw "Refusing cleanup outside the exact investor-demo runtime directory."
    }
    if (Test-Path -LiteralPath $resolvedRunRoot -PathType Container) {
        Remove-Item -LiteralPath $resolvedRunRoot -Recurse -Force
    }
}

function Get-CimProcessStartUtc([object]$Value, [string]$Label) {
    if ($null -eq $Value) { throw "$Label creation time is unavailable." }
    if ($Value -is [DateTime]) { return ([DateTime]$Value).ToUniversalTime() }
    if ($Value -is [DateTimeOffset]) { return ([DateTimeOffset]$Value).UtcDateTime }
    try { return [DateTimeOffset]::Parse([string]$Value).UtcDateTime }
    catch {
        try {
            return [Management.ManagementDateTimeConverter]::ToDateTime(
                [string]$Value
            ).ToUniversalTime()
        }
        catch { throw "$Label creation time is invalid." }
    }
}

function Test-StartedProcessSnapshot([object]$Process, [object]$Snapshot) {
    if ($null -eq $Process) { return $false }
    return (
        $Process.Id -eq [int]$Snapshot.ProcessId -and
        [string]::Equals(
            [string]$Process.ProcessName,
            [string]$Snapshot.ProcessName,
            [StringComparison]::OrdinalIgnoreCase
        ) -and
        [Math]::Abs(
            ($Process.StartTime.ToUniversalTime() - $Snapshot.StartedAtUtc).TotalSeconds
        ) -lt 1.0
    )
}

function Stop-StartedProcessTree([System.Diagnostics.Process]$RootProcess) {
    if ($null -eq $RootProcess) { return }
    $rootId = $RootProcess.Id
    $rootSnapshot = [pscustomobject]@{
        ProcessId = $rootId
        ParentProcessId = 0
        ProcessName = [string]$RootProcess.ProcessName
        StartedAtUtc = $RootProcess.StartTime.ToUniversalTime()
        EndedAtUtc = if ($RootProcess.HasExited) {
            $RootProcess.ExitTime.ToUniversalTime()
        }
        else { $null }
        Depth = 0
    }
    $currentRoot = Get-Process -Id $rootId -ErrorAction SilentlyContinue
    if ($null -ne $currentRoot -and -not (Test-StartedProcessSnapshot $currentRoot $rootSnapshot)) {
        throw "Started launcher PID $rootId was recycled; cleanup was not attempted."
    }

    $processTable = @(
        Get-CimInstance -ClassName Win32_Process -ErrorAction Stop |
            Select-Object ProcessId, ParentProcessId, CreationDate
    )
    if ($processTable.Count -eq 0) {
        throw "Started process-tree metadata is unavailable; cleanup was not attempted."
    }
    # Bind the launcher snapshot to the same CIM table used to infer descendants.
    $rootEntries = @(
        $processTable | Where-Object { [int]$_.ProcessId -eq $rootId }
    )
    $currentRoot = Get-Process -Id $rootId -ErrorAction SilentlyContinue
    if ($null -ne $currentRoot) {
        if ($rootEntries.Count -ne 1) {
            throw "Started launcher PID $rootId has ambiguous process-tree metadata; cleanup was not attempted."
        }
        $cimRootStartedAtUtc = Get-CimProcessStartUtc `
            $rootEntries[0].CreationDate "Started launcher PID $rootId"
        if (
            [Math]::Abs(($rootSnapshot.StartedAtUtc - $cimRootStartedAtUtc).TotalSeconds) -ge 1.0 -or
            -not (Test-StartedProcessSnapshot $currentRoot $rootSnapshot)
        ) {
            throw "Started launcher PID $rootId was recycled during process-tree capture; cleanup was not attempted."
        }
    }
    elseif ($null -eq $rootSnapshot.EndedAtUtc) {
        throw "Started launcher PID $rootId exited during process-tree capture; runtime state was retained."
    }
    elseif ($rootEntries.Count -ne 0) {
        throw "Started launcher PID $rootId was recycled during process-tree capture; cleanup was not attempted."
    }
    $snapshotById = @{}
    $snapshotById[$rootId] = $rootSnapshot
    $knownIds = [Collections.Generic.HashSet[int]]::new()
    [void]$knownIds.Add($rootId)
    $changed = $true
    while ($changed) {
        $changed = $false
        foreach ($entry in $processTable) {
            $processId = [int]$entry.ProcessId
            $parentId = [int]$entry.ParentProcessId
            if (
                $processId -le 0 -or
                $processId -eq $parentId -or
                $knownIds.Contains($processId) -or
                -not $knownIds.Contains($parentId)
            ) {
                continue
            }
            $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
            if ($null -eq $process) { continue }
            $startedAtUtc = $process.StartTime.ToUniversalTime()
            $cimStartedAtUtc = Get-CimProcessStartUtc `
                $entry.CreationDate "Started descendant PID $processId"
            if ([Math]::Abs(($startedAtUtc - $cimStartedAtUtc).TotalSeconds) -ge 1.0) {
                throw "Started descendant PID $processId was recycled; cleanup was not attempted."
            }
            $parentSnapshot = $snapshotById[$parentId]
            if (($startedAtUtc - $parentSnapshot.StartedAtUtc).TotalSeconds -lt -1.0) {
                # Windows retains ParentProcessId after exit; an older process cannot
                # belong to this verified parent after its PID has been reused.
                continue
            }
            if (
                $null -ne $parentSnapshot.EndedAtUtc -and
                ($startedAtUtc - $parentSnapshot.EndedAtUtc).TotalSeconds -gt 1.0
            ) {
                throw "Started descendant PID $processId postdates its verified parent; cleanup was not attempted."
            }
            $snapshotById[$processId] = [pscustomobject]@{
                ProcessId = $processId
                ParentProcessId = $parentId
                ProcessName = [string]$process.ProcessName
                StartedAtUtc = $startedAtUtc
                EndedAtUtc = $null
                Depth = [int]$parentSnapshot.Depth + 1
            }
            [void]$knownIds.Add($processId)
            $changed = $true
        }
    }

    $snapshots = @($snapshotById.Values | Sort-Object Depth -Descending)
    # Revalidate the launcher immediately before any inferred descendant is stopped.
    $currentRoot = Get-Process -Id $rootId -ErrorAction SilentlyContinue
    if ($null -eq $currentRoot) {
        if ($null -eq $rootSnapshot.EndedAtUtc) {
            throw "Started launcher PID $rootId exited during cleanup enumeration; runtime state was retained."
        }
    }
    elseif (-not (Test-StartedProcessSnapshot $currentRoot $rootSnapshot)) {
        throw "Started launcher PID $rootId was recycled during cleanup enumeration; runtime state was retained."
    }
    foreach ($snapshot in $snapshots) {
        $process = Get-Process -Id ([int]$snapshot.ProcessId) -ErrorAction SilentlyContinue
        if ($null -eq $process) { continue }
        if (-not (Test-StartedProcessSnapshot $process $snapshot)) {
            throw "Started process PID $($snapshot.ProcessId) was recycled; it was not stopped."
        }
        Stop-Process -InputObject $process -Force -ErrorAction SilentlyContinue
        $process.WaitForExit(5000) | Out-Null
    }
    foreach ($snapshot in $snapshots) {
        $processId = [int]$snapshot.ProcessId
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($null -eq $process) { continue }
        if (Test-StartedProcessSnapshot $process $snapshot) {
            throw "Started process PID $processId remained alive; runtime state was retained."
        }
        Write-Warning "PID $processId was recycled after its started process exited; the new process was not touched."
    }
}

function Assert-NoStartedDemoListeners {
    foreach ($port in @($workerPort, $ApiPort, $WebPort) | Select-Object -Unique) {
        if (@(Get-ListeningConnections $port).Count -gt 0) {
            throw "Demo cleanup left a listener on port $port."
        }
    }
}

function Remove-InvestorDemoProcessVariable([hashtable]$Snapshot, [string]$Name) {
    if (-not $Snapshot.ContainsKey($Name)) {
        $existing = Get-Item -LiteralPath "Env:$Name" -ErrorAction SilentlyContinue
        $Snapshot[$Name] = [pscustomobject]@{
            Exists = $null -ne $existing
            Value = if ($null -eq $existing) { $null } else { $existing.Value }
        }
    }
    Remove-Item -LiteralPath "Env:$Name" -ErrorAction SilentlyContinue
}

function Set-InvestorDemoEnvironment([object]$Runtime, [hashtable]$Snapshot) {
    $runtimeSnapshot = Set-AtlasLensRuntimeEnvironment -Configuration $Runtime -IncludeCaches -IncludeOfflineModelGuards
    foreach ($entry in $runtimeSnapshot.GetEnumerator()) { $Snapshot[$entry.Key] = $entry.Value }
    $values = [ordered]@{
        APP_ENV = "development"
        API_HOST = "127.0.0.1"
        API_PORT = [string]$ApiPort
        ALLOWED_ORIGINS = "http://127.0.0.1:$WebPort,http://localhost:$WebPort"
        DATABASE_URL = $databaseUrl
        TEMP_STORAGE_DIR = $temporaryStorage
        KEEP_UPLOADS = "false"
        GLOBAL_MODEL_ENABLED = "false"
        CUSTOM_MODEL_ENABLED = "false"
        OCR_ENABLED = "false"
        RETRIEVAL_ENABLED = "false"
        PHASE6A_ENABLED = "false"
        PHASE6B_ENABLED = "false"
        PHASE6C_ENABLED = "false"
        OPENAI_GEO_ENABLED = "false"
        NVIDIA_VISION_ENABLED = "false"
        MAPILLARY_ENABLED = "false"
        REFERENCE_INDEX_ENABLED = "false"
        TURKIYE_REFERENCE_INDEX_ENABLED = "false"
        MEGALOC_WORKER_ENABLED = "true"
        MEGALOC_WORKER_HOST = "127.0.0.1"
        MEGALOC_WORKER_PORT = [string]$workerPort
        MEGALOC_DEVICE = $MegaLocDevice
        MEGALOC_TIMEOUT_SECONDS = "180"
        ATLASLENS_TURKIYE_DEMO_ENABLED = "true"
        ATLASLENS_TURKIYE_DEMO_BUNDLE_PATH = $bundlePath
        ATLASLENS_TURKIYE_DEMO_EXPECTED_PUBLICATION_SHA256 = $publicationSha256
        ATLASLENS_TURKIYE_DEMO_EXPECTED_SOURCE_POLICY_SHA256 = $sourcePolicySha256
        ATLASLENS_TURKIYE_DEMO_EXPECTED_SELECTION_LOCK_SHA256 = $selectionLockSha256
        ATLASLENS_TURKIYE_DEMO_TOP_K = "5"
        ATLASLENS_TURKIYE_DEMO_UNCERTAINTY_RADIUS_M = "1000"
        HF_DATASETS_OFFLINE = "1"
        HF_HUB_DISABLE_TELEMETRY = "1"
        HF_HUB_OFFLINE = "1"
        TRANSFORMERS_OFFLINE = "1"
        UV_OFFLINE = "1"
        npm_config_offline = "true"
        VITE_MAP_PROVIDER = "maplibre"
        VITE_MAP_TILE_URL = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
        ATLASLENS_IGNORE_ENV_FILE = "true"
    }
    foreach ($entry in $values.GetEnumerator()) {
        Set-AtlasLensProcessVariable -Snapshot $Snapshot -Name $entry.Key -Value $entry.Value
    }
    foreach ($name in @(
        "NVIDIA_API_KEY",
        "OPENAI_API_KEY",
        "MAPILLARY_ACCESS_TOKEN",
        "VITE_MAP_STYLE_URL"
    )) {
        Remove-InvestorDemoProcessVariable -Snapshot $Snapshot -Name $name
    }
}

# All fail-closed gates run before creating runtime state or starting a process.
Assert-AtlasLensDiskBudget
if ([string]::IsNullOrWhiteSpace($RuntimeRoot)) {
    throw "Investor demo startup requires an explicit external RuntimeRoot so project env files cannot be read."
}
$runtime = Get-AtlasLensRuntimeConfiguration -RepositoryRoot $repoRoot -RuntimeRoot $RuntimeRoot
if (-not $runtime.ExternalRuntime) { throw "Investor demo startup requires an external RuntimeRoot." }
$preflightSnapshot = Set-AtlasLensRuntimeEnvironment -Configuration $runtime -IncludeCaches -IncludeOfflineModelGuards
try { Assert-AtlasLensApiRuntime $runtime }
finally { Restore-AtlasLensRuntimeEnvironment $preflightSnapshot }
if ($null -eq (Get-Command "npm.cmd" -ErrorAction SilentlyContinue)) {
    throw "Required command 'npm.cmd' was not found on PATH."
}
if (-not (Test-Path -LiteralPath (Join-Path $runtime.WebDirectory "node_modules\vite\package.json") -PathType Leaf)) {
    throw "Frontend dependencies are missing; startup will not download them."
}
foreach ($required in @(
    $workerStartScript,
    $workerStopScript,
    $apiScript,
    $webScript,
    $bundlePath,
    $pilotRoot,
    (Join-Path $repoRoot ".local\workers\megaloc\.venv\Scripts\python.exe"),
    (Join-Path $repoRoot "services\model-workers\megaloc\worker.py")
)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required private-demo runtime component is missing: $required" }
}
Assert-InvestorDemoRuntimePath
if (Test-Path -LiteralPath $runRoot) {
    if (Test-Path -LiteralPath $statePath -PathType Leaf) {
        throw "Investor demo state already exists. Run stop-investor-demo.ps1 before starting another instance."
    }
    throw "An untrusted investor-demo runtime directory already exists; it was not reused."
}
foreach ($portGate in @(
    [pscustomobject]@{ Port = $workerPort; Label = "MegaLoc worker" },
    [pscustomobject]@{ Port = $ApiPort; Label = "API" },
    [pscustomobject]@{ Port = $WebPort; Label = "Web" }
)) {
    Assert-PortFree $portGate.Port $portGate.Label
}

New-Item -ItemType Directory -Force -Path $runRoot, $logRoot, $dataRoot, $temporaryStorage | Out-Null
Assert-InvestorDemoRuntimePath
try {
    Set-InvestorDemoEnvironment -Runtime $runtime -Snapshot $environmentSnapshot
    & $powershell -NoProfile -ExecutionPolicy Bypass -File $workerStartScript `
        -StartupTimeoutSeconds $StartupTimeoutSeconds -Device $MegaLocDevice
    if ($LASTEXITCODE -ne 0) { throw "MegaLoc worker startup failed with code $LASTEXITCODE." }
    $workerStarted = $true
    $workerRecord = Get-TrustedWorkerRecord
    Write-DemoState "starting"

    Push-Location $dataRoot
    try {
        $prepareOutput = @(& $runtime.ApiPython -m atlaslens_api.cli private-demo prepare `
            --database-url $databaseUrl `
            --expected-publication-sha256 $publicationSha256 `
            --expected-source-policy-sha256 $sourcePolicySha256 `
            --expected-selection-lock-sha256 $selectionLockSha256 `
            --pilot-root $pilotRoot `
            --worker-host 127.0.0.1 `
            --worker-port $workerPort `
            --device $MegaLocDevice)
        $prepareExitCode = $LASTEXITCODE
        if ($prepareExitCode -ne 0) {
            throw "Private-demo case preparation failed with code $prepareExitCode."
        }
        $prepareLines = @(
            $prepareOutput |
                ForEach-Object { [string]$_ } |
                Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
        )
        if ($prepareLines.Count -ne 1) {
            throw "Private-demo preparation returned an invalid receipt."
        }
        try { $prepareReceipt = $prepareLines[0] | ConvertFrom-Json }
        catch { throw "Private-demo preparation returned an invalid receipt." }
        if (
            $prepareReceipt.status -notin @("created", "existing", "repaired") -or
            $prepareReceipt.audit_integrity_valid -ne $true -or
            $prepareReceipt.analysis_mode -ne "local_only" -or
            $prepareReceipt.query_bytes_retained -ne $false
        ) {
            throw "Private-demo preparation receipt failed its safety contract."
        }
        $demoCaseId = ConvertTo-CanonicalDemoCaseId $prepareReceipt.case_id
        Write-DemoState "starting"
    }
    finally { Pop-Location }

    $apiArguments = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Quote-ProcessArgument $apiScript),
        "-ApiHost", "127.0.0.1", "-ApiPort", [string]$ApiPort,
        "-RuntimeRoot", (Quote-ProcessArgument $runtime.RuntimeRoot),
        # The private-demo CLI just created this disposable DB from current metadata;
        # running Alembic against its intentionally unversioned tables would collide.
        "-IgnoreProjectEnv", "-SkipMigration"
    )
    $apiProcess = Start-Process -FilePath $powershell -ArgumentList $apiArguments `
        -PassThru -WindowStyle Hidden -RedirectStandardOutput $apiLogOut -RedirectStandardError $apiLogErr
    Wait-JsonEndpoint -Url "http://127.0.0.1:$ApiPort/api/v1/health" -Process $apiProcess -Label "API" -Validate {
        param($body) $body.status -eq "ok" -and -not [string]::IsNullOrWhiteSpace($body.version)
    }
    Wait-JsonEndpoint -Url "http://127.0.0.1:$ApiPort/api/v1/ready" -Process $apiProcess -Label "API" -Validate {
        param($body) $body.status -eq "ready"
    }
    Wait-JsonEndpoint -Url "http://127.0.0.1:$ApiPort/api/v1/mapillary-demo/status" -Process $apiProcess -Label "Private demo" -Validate {
        param($body) $body.state -eq "active" -and $body.available -eq $true
    }
    Wait-JsonEndpoint -Url "http://127.0.0.1:$ApiPort/api/v1/cases/$demoCaseId" -Process $apiProcess -Label "Investor demo case" -Validate {
        param($body) [string]$body.id -eq $demoCaseId
    }
    $apiRecord = New-ServiceRecord -Name "api" -Launcher $apiProcess -Port $ApiPort -ScriptPath $apiScript
    Write-DemoState "starting"

    $webArguments = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Quote-ProcessArgument $webScript),
        "-WebHost", "127.0.0.1", "-WebPort", [string]$WebPort,
        "-ApiTarget", "http://127.0.0.1:$ApiPort",
        "-RuntimeRoot", (Quote-ProcessArgument $runtime.RuntimeRoot), "-IgnoreProjectEnv"
    )
    $webProcess = Start-Process -FilePath $powershell -ArgumentList $webArguments `
        -PassThru -WindowStyle Hidden -RedirectStandardOutput $webLogOut -RedirectStandardError $webLogErr
    Wait-WebHttp200 -Url "http://127.0.0.1:$WebPort/" -Process $webProcess
    Wait-JsonEndpoint -Url "http://127.0.0.1:$WebPort/api/v1/mapillary-demo/status" -Process $webProcess -Label "Web API proxy" -Validate {
        param($body) $body.state -eq "active" -and $body.available -eq $true
    }
    Wait-JsonEndpoint -Url "http://127.0.0.1:$WebPort/api/v1/cases/$demoCaseId" -Process $webProcess -Label "Web investor case" -Validate {
        param($body) [string]$body.id -eq $demoCaseId
    }
    foreach ($pagePath in @("media", "evidence", "hypotheses", "audit-events")) {
        Wait-JsonEndpoint `
            -Url "http://127.0.0.1:$WebPort/api/v1/cases/$demoCaseId/$pagePath`?limit=100&offset=0" `
            -Process $webProcess -Label "Web investor case $pagePath" -Validate {
                param($body)
                $null -ne $body.PSObject.Properties["items"] -and
                $null -ne $body.PSObject.Properties["total"]
            }
    }
    Wait-JsonEndpoint -Url "http://127.0.0.1:$WebPort/api/v1/cases/$demoCaseId/audit-integrity" -Process $webProcess -Label "Web investor audit integrity" -Validate {
        param($body) $body.valid -eq $true
    }
    $webRecord = New-ServiceRecord -Name "web" -Launcher $webProcess -Port $WebPort -ScriptPath $webScript
    Write-DemoState "ready"
    $startupComplete = $true

    $demoUrl = "http://127.0.0.1:$WebPort/?demo=investor&lang=tr&caseId=$([Uri]::EscapeDataString($demoCaseId))"

    Write-Host "AtlasLens investor demo is ready (local-only; downloads and cloud providers disabled)."
    Write-Host "Demo: $demoUrl"
    Write-Host "Stop: powershell -NoProfile -ExecutionPolicy Bypass -File `"$PSScriptRoot\stop-investor-demo.ps1`""
}
finally {
    if ($startupComplete) {
        Restore-AtlasLensRuntimeEnvironment $environmentSnapshot
    }
    else {
        $cleanupFailed = $false
        try { Restore-AtlasLensRuntimeEnvironment $environmentSnapshot }
        catch {
            $cleanupFailed = $true
            Write-Warning "Investor-demo environment restoration failed; runtime state will be retained."
        }
        $serviceCleanupFailed = $false
        foreach ($startedService in @(
            [pscustomobject]@{ Label = "web"; Process = $webProcess },
            [pscustomobject]@{ Label = "api"; Process = $apiProcess }
        )) {
            try { Stop-StartedProcessTree $startedService.Process }
            catch {
                $serviceCleanupFailed = $true
                $cleanupFailed = $true
                Write-Warning "Started $($startedService.Label) cleanup failed; runtime state will be retained."
            }
        }
        if (-not $serviceCleanupFailed) {
            try {
                foreach ($port in @($ApiPort, $WebPort) | Select-Object -Unique) {
                    if (@(Get-ListeningConnections $port).Count -gt 0) {
                        throw "Started service cleanup left a listener on port $port."
                    }
                }
            }
            catch {
                $serviceCleanupFailed = $true
                $cleanupFailed = $true
                Write-Warning "Started service listener verification failed; runtime state will be retained."
            }
        }
        if ($workerStarted -and -not $cleanupFailed) {
            try {
                & $powershell -NoProfile -ExecutionPolicy Bypass -File $workerStopScript
                $workerStopExitCode = $LASTEXITCODE
                if ($workerStopExitCode -ne 0) {
                    throw "MegaLoc cleanup failed with code $workerStopExitCode."
                }
            }
            catch {
                $cleanupFailed = $true
                Write-Warning "MegaLoc cleanup failed; trusted runtime state was retained."
            }
        }
        elseif ($workerStarted) {
            Write-Warning "MegaLoc cleanup was skipped so its trusted metadata remains available for recovery."
        }
        if (-not $cleanupFailed) {
            try { Assert-NoStartedDemoListeners }
            catch {
                $cleanupFailed = $true
                Write-Warning "Final demo listener verification failed; runtime state was retained."
            }
        }
        if (-not $cleanupFailed) {
            try { Remove-InvestorDemoRuntime }
            catch {
                $cleanupFailed = $true
                Write-Warning "Investor-demo runtime cleanup failed; runtime state was retained."
            }
        }
    }
}
