[CmdletBinding()]
param(
    [ValidateRange(1, 30)]
    [int]$StopTimeoutSeconds = 10
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$runRoot = Join-Path $repoRoot ".local\run\investor-demo"
$statePath = Join-Path $runRoot "state.json"
$workerMetadataPath = Join-Path $repoRoot ".local\run\phase6c-workers\megaloc.json"
$apiScript = Join-Path $PSScriptRoot "start-api.ps1"
$webScript = Join-Path $PSScriptRoot "start-web.ps1"
$workerPath = Join-Path $repoRoot "services\model-workers\megaloc\worker.py"
$powershell = (Get-Command "powershell.exe" -ErrorAction Stop).Source
$taskkill = (Get-Command "taskkill.exe" -ErrorAction Stop).Source
$workerPort = 8794
$listenerStartMaximumDelaySeconds = 30.0

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

function Assert-InvestorDemoStateFileSafe {
    $expectedStatePath = [IO.Path]::GetFullPath(
        (Join-Path $repoRoot ".local\run\investor-demo\state.json")
    )
    $resolvedStatePath = [IO.Path]::GetFullPath($statePath)
    if (-not [string]::Equals($resolvedStatePath, $expectedStatePath, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Investor-demo state path is not the exact expected file; no process was stopped."
    }
    if (-not (Test-Path -LiteralPath $resolvedStatePath -PathType Leaf)) {
        throw "Investor-demo state is not a regular file; no process was stopped."
    }
    $item = Get-Item -LiteralPath $resolvedStatePath -Force
    if (
        $item.PSIsContainer -or
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
    ) {
        throw "Investor-demo state is not a regular non-reparse file; no process was stopped."
    }
}

function Get-PropertyValue([object]$Value, [string]$Name) {
    if ($null -eq $Value) { return $null }
    $property = $Value.PSObject.Properties[$Name]
    if ($null -eq $property) { return $null }
    return $property.Value
}

function Get-ListeningConnections([int]$Port) {
    return @(
        Get-NetTCPConnection -State Listen -ErrorAction Stop |
            Where-Object { [int]$_.LocalPort -eq $Port }
    )
}

function Get-LoopbackPortOwner([int]$Port) {
    $listeners = @(Get-ListeningConnections $Port)
    if ($listeners.Count -eq 0) { return $null }
    if (@($listeners | Where-Object { $_.LocalAddress -ne "127.0.0.1" }).Count -gt 0) {
        throw "Port $Port is no longer bound exclusively to 127.0.0.1; no process was stopped."
    }
    $owners = @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)
    if ($owners.Count -ne 1) { throw "Port $Port has ambiguous ownership; no process was stopped." }
    return [int]$owners[0]
}

function Assert-NoInvestorDemoListeners([object]$State) {
    $ports = @($workerPort)
    foreach ($service in @($State.api, $State.web)) {
        if ($null -ne $service) { $ports += [int](Get-PropertyValue $service "port") }
    }
    foreach ($port in $ports | Select-Object -Unique) {
        $listeners = @(Get-ListeningConnections $port)
        if ($listeners.Count -gt 0) {
            throw "Port $port still has a listener; trusted metadata and runtime state were retained."
        }
    }
}

function Get-ProcessCommand([int]$ProcessId) {
    return Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
}

function Test-RecordedStart([object]$Process, [string]$RecordedValue) {
    try {
        $recorded = [DateTimeOffset]::Parse($RecordedValue).UtcDateTime
        return [Math]::Abs(($recorded - $Process.StartTime.ToUniversalTime()).TotalSeconds) -lt 1.0
    }
    catch { return $false }
}

function Get-RecordedStartUtc([string]$RecordedValue, [string]$Label) {
    try { return [DateTimeOffset]::Parse($RecordedValue).UtcDateTime }
    catch { throw "$Label recorded start time is invalid; no process was stopped." }
}

function Assert-RecordedProcessIdentity(
    [object]$Process,
    [int]$ExpectedProcessId,
    [string]$RecordedStart,
    [string]$ExpectedProcessName,
    [string]$Label
) {
    if (
        $null -eq $Process -or
        $Process.Id -ne $ExpectedProcessId -or
        -not (Test-RecordedStart $Process $RecordedStart) -or
        -not [string]::Equals(
            [string]$Process.ProcessName,
            $ExpectedProcessName,
            [StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw "$Label PID, start time, or process name does not match trusted state; no process was stopped."
    }
}

function Assert-OptionalCommandMetadata(
    [AllowNull()][object]$Command,
    [AllowEmptyString()][string]$ExpectedExecutablePath,
    [AllowEmptyString()][string]$ExpectedCommandMarker,
    [string]$ExpectedProcessName,
    [string]$Label
) {
    $fallbackRequired = $false
    if ($null -eq $Command) { return $true }

    $executablePath = [string](Get-PropertyValue $Command "ExecutablePath")
    if ([string]::IsNullOrWhiteSpace($executablePath)) {
        $fallbackRequired = $true
    }
    elseif (-not [string]::IsNullOrWhiteSpace($ExpectedExecutablePath)) {
        if (-not [string]::Equals($executablePath, $ExpectedExecutablePath, [StringComparison]::OrdinalIgnoreCase)) {
            throw "$Label has nonblank mismatching executable metadata; no process was stopped."
        }
    }
    else {
        $executableName = [IO.Path]::GetFileNameWithoutExtension($executablePath)
        if (-not [string]::Equals($executableName, $ExpectedProcessName, [StringComparison]::OrdinalIgnoreCase)) {
            throw "$Label has nonblank mismatching executable metadata; no process was stopped."
        }
    }

    $commandLine = [string](Get-PropertyValue $Command "CommandLine")
    if ([string]::IsNullOrWhiteSpace($commandLine)) {
        $fallbackRequired = $true
    }
    elseif (
        -not [string]::IsNullOrWhiteSpace($ExpectedCommandMarker) -and
        $commandLine.Replace("/", "\").IndexOf(
            $ExpectedCommandMarker.Replace("/", "\"),
            [StringComparison]::OrdinalIgnoreCase
        ) -lt 0
    ) {
        throw "$Label has nonblank mismatching command-line metadata; no process was stopped."
    }
    return $fallbackRequired
}

function Assert-ExpectedPortOwner([AllowNull()][object]$ActualOwner, [int]$ExpectedProcessId, [string]$Label) {
    if ($null -eq $ActualOwner -or [int]$ActualOwner -ne $ExpectedProcessId) {
        $description = if ($null -eq $ActualOwner) { "none" } else { [string]$ActualOwner }
        throw "$Label loopback port belongs to $description instead of trusted PID $ExpectedProcessId; no process was stopped."
    }
}

function Assert-ListenerStartRelationship(
    [DateTime]$LauncherStartedAtUtc,
    [DateTime]$ListenerStartedAtUtc,
    [string]$Label
) {
    $delay = ($ListenerStartedAtUtc.ToUniversalTime() - $LauncherStartedAtUtc.ToUniversalTime()).TotalSeconds
    if ($delay -lt -1.0 -or $delay -gt $listenerStartMaximumDelaySeconds) {
        throw "$Label listener start time is not after or near its trusted launcher; no process was stopped."
    }
}

function Assert-ServiceOwnership(
    [object]$Record,
    [string]$ExpectedName,
    [string]$ExpectedScript,
    [string]$ListenerMarker,
    [string]$ExpectedListenerProcessName
) {
    if ($null -eq $Record) { return }
    if (
        (Get-PropertyValue $Record "name") -ne $ExpectedName -or
        -not [string]::Equals(
            [string](Get-PropertyValue $Record "script_path"),
            $ExpectedScript,
            [StringComparison]::OrdinalIgnoreCase
        ) -or
        -not [string]::Equals(
            [string](Get-PropertyValue $Record "launcher_executable"),
            $powershell,
            [StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw "$ExpectedName state identity is invalid; no process was stopped."
    }
    $port = [int](Get-PropertyValue $Record "port")
    if ($port -lt 1 -or $port -gt 65535 -or $port -eq $workerPort) {
        throw "$ExpectedName state port is invalid; no process was stopped."
    }
    $launcherId = [int](Get-PropertyValue $Record "launcher_pid")
    $listenerId = [int](Get-PropertyValue $Record "listener_pid")
    if ($launcherId -le 0 -or $listenerId -le 0 -or $launcherId -eq $listenerId) {
        throw "$ExpectedName state PIDs are invalid; no process was stopped."
    }
    $launcherRecordedStart = Get-RecordedStartUtc `
        ([string](Get-PropertyValue $Record "launcher_started_at_utc")) "$ExpectedName launcher"
    $listenerRecordedStart = Get-RecordedStartUtc `
        ([string](Get-PropertyValue $Record "listener_started_at_utc")) "$ExpectedName listener"
    Assert-ListenerStartRelationship $launcherRecordedStart $listenerRecordedStart $ExpectedName
    $launcher = Get-Process -Id $launcherId -ErrorAction SilentlyContinue
    $listener = Get-Process -Id $listenerId -ErrorAction SilentlyContinue

    if ($null -ne $launcher) {
        Assert-RecordedProcessIdentity $launcher $launcherId `
            ([string](Get-PropertyValue $Record "launcher_started_at_utc")) "powershell" "$ExpectedName launcher"
        $command = Get-ProcessCommand $launcherId
        [void](Assert-OptionalCommandMetadata $command $powershell $ExpectedScript "powershell" "$ExpectedName launcher")
    }

    if ($null -ne $listener) {
        Assert-RecordedProcessIdentity $listener $listenerId `
            ([string](Get-PropertyValue $Record "listener_started_at_utc")) `
            $ExpectedListenerProcessName "$ExpectedName listener"
        $command = Get-ProcessCommand $listenerId
        $owner = Get-LoopbackPortOwner $port
        Assert-ExpectedPortOwner $owner $listenerId "$ExpectedName listener"
        [void](Assert-OptionalCommandMetadata `
            $command "" $ListenerMarker $ExpectedListenerProcessName "$ExpectedName listener"
        )
        $launcherStart = if ($null -eq $launcher) {
            $launcherRecordedStart
        }
        else { $launcher.StartTime.ToUniversalTime() }
        Assert-ListenerStartRelationship $launcherStart $listener.StartTime.ToUniversalTime() $ExpectedName
    }

    if ($null -eq $listener) {
        $currentOwner = Get-LoopbackPortOwner $port
        if ($null -ne $currentOwner) {
            throw "$ExpectedName port is owned by foreign PID $currentOwner; no process was stopped."
        }
    }
}

function Assert-WorkerMetadataPathSafe {
    $expectedPath = [IO.Path]::GetFullPath(
        (Join-Path $repoRoot ".local\run\phase6c-workers\megaloc.json")
    )
    $resolvedPath = [IO.Path]::GetFullPath($workerMetadataPath)
    if (-not [string]::Equals($resolvedPath, $expectedPath, [StringComparison]::OrdinalIgnoreCase)) {
        throw "MegaLoc PID metadata path is not the exact expected file; no process was stopped."
    }
    foreach ($candidate in @(
        (Join-Path $repoRoot ".local"),
        (Join-Path $repoRoot ".local\run"),
        (Join-Path $repoRoot ".local\run\phase6c-workers")
    )) {
        if (-not (Test-Path -LiteralPath $candidate)) { continue }
        $item = Get-Item -LiteralPath $candidate -Force
        if (
            -not $item.PSIsContainer -or
            ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
        ) {
            throw "MegaLoc PID metadata path contains a file or reparse point; no process was stopped."
        }
    }
    if (Test-Path -LiteralPath $resolvedPath) {
        $item = Get-Item -LiteralPath $resolvedPath -Force
        if (
            $item.PSIsContainer -or
            ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
        ) {
            throw "MegaLoc PID metadata is not a regular file; no process was stopped."
        }
    }
}

function Assert-WorkerOwnership([object]$Record) {
    if ($null -eq $Record) { return }
    if (
        (Get-PropertyValue $Record "provider") -ne "megaloc" -or
        [int](Get-PropertyValue $Record "port") -ne $workerPort -or
        -not [string]::Equals(
            [string](Get-PropertyValue $Record "worker_path"),
            $workerPath,
            [StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw "MegaLoc investor-demo state is invalid; no process was stopped."
    }
    $processId = [int](Get-PropertyValue $Record "pid")
    if ($processId -le 0) {
        throw "MegaLoc investor-demo PID is invalid; no process was stopped."
    }
    Assert-WorkerMetadataPathSafe
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    $metadata = $null
    if (Test-Path -LiteralPath $workerMetadataPath -PathType Leaf) {
        try { $metadata = Get-Content -LiteralPath $workerMetadataPath -Raw | ConvertFrom-Json }
        catch { throw "MegaLoc PID metadata is invalid; no process was stopped." }
        if (
            $metadata.schema_version -ne "atlaslens-worker-process-v1" -or
            $metadata.provider -ne "megaloc" -or
            [int]$metadata.pid -ne $processId -or
            [int]$metadata.port -ne $workerPort -or
            -not [string]::Equals(
                [string]$metadata.worker_path,
                $workerPath,
                [StringComparison]::OrdinalIgnoreCase
            )
        ) {
            throw "MegaLoc PID metadata is not owned by this investor demo; no process was stopped."
        }
        $investorRecordedStart = Get-RecordedStartUtc `
            ([string](Get-PropertyValue $Record "process_started_at_utc")) `
            "MegaLoc investor-demo state"
        $metadataRecordedStart = Get-RecordedStartUtc `
            ([string]$metadata.process_started_at_utc) "MegaLoc PID metadata"
        if ([Math]::Abs(($investorRecordedStart - $metadataRecordedStart).TotalSeconds) -ge 1.0) {
            throw "MegaLoc trusted records have different process start times; no process was stopped."
        }
    }
    if ($null -eq $process) {
        $owner = Get-LoopbackPortOwner $workerPort
        if ($null -ne $owner) {
            throw "MegaLoc state PID exited but port $workerPort belongs to PID $owner; no process was stopped."
        }
        return
    }
    if ($null -eq $metadata) {
        throw "MegaLoc PID metadata is missing; no process was stopped."
    }
    Assert-RecordedProcessIdentity $process $processId `
        ([string](Get-PropertyValue $Record "process_started_at_utc")) "python" "MegaLoc worker"
    Assert-RecordedProcessIdentity $process $processId `
        ([string]$metadata.process_started_at_utc) "python" "MegaLoc worker metadata"
    $command = Get-ProcessCommand $processId
    $owner = Get-LoopbackPortOwner $workerPort
    Assert-ExpectedPortOwner $owner $processId "MegaLoc worker"
    [void](Assert-OptionalCommandMetadata $command "" $workerPath "python" "MegaLoc worker")
}

function Get-CimProcessStartUtc([object]$Value, [string]$Label) {
    if ($null -eq $Value) {
        throw "$Label creation time is unavailable; no process was stopped."
    }
    if ($Value -is [DateTime]) { return ([DateTime]$Value).ToUniversalTime() }
    if ($Value -is [DateTimeOffset]) { return ([DateTimeOffset]$Value).UtcDateTime }
    try { return [DateTimeOffset]::Parse([string]$Value).UtcDateTime }
    catch {
        try {
            return [Management.ManagementDateTimeConverter]::ToDateTime(
                [string]$Value
            ).ToUniversalTime()
        }
        catch { throw "$Label creation time is invalid; no process was stopped." }
    }
}

function Assert-ProcessMatchesSnapshot([object]$Process, [object]$Snapshot, [string]$Label) {
    Assert-RecordedProcessIdentity `
        $Process ([int]$Snapshot.ProcessId) ([string]$Snapshot.RecordedStart) `
        ([string]$Snapshot.ProcessName) $Label
}

function Assert-OwnedRootSnapshotsCurrent([object[]]$Snapshots, [string]$ExpectedName) {
    foreach ($snapshot in @($Snapshots | Where-Object { $_.Role -in @("launcher", "listener") })) {
        $processId = [int]$snapshot.ProcessId
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($null -eq $process) {
            throw "$ExpectedName $($snapshot.Role) PID $processId exited during cleanup enumeration; no process was stopped."
        }
        Assert-ProcessMatchesSnapshot `
            $process $snapshot "$ExpectedName $($snapshot.Role) pre-stop"
    }
}

function Get-OwnedProcessSnapshot(
    [object]$Record,
    [string]$ExpectedListenerProcessName,
    [string]$ExpectedName
) {
    if ($null -eq $Record) { return @() }
    $launcherId = [int](Get-PropertyValue $Record "launcher_pid")
    $listenerId = [int](Get-PropertyValue $Record "listener_pid")
    $snapshotById = @{}
    foreach ($root in @(
        [pscustomobject]@{
            ProcessId = $launcherId
            Role = "launcher"
            RecordedStart = [string](Get-PropertyValue $Record "launcher_started_at_utc")
            ProcessName = "powershell"
        },
        [pscustomobject]@{
            ProcessId = $listenerId
            Role = "listener"
            RecordedStart = [string](Get-PropertyValue $Record "listener_started_at_utc")
            ProcessName = $ExpectedListenerProcessName
        }
    )) {
        $process = Get-Process -Id $root.ProcessId -ErrorAction SilentlyContinue
        if ($null -eq $process) { continue }
        Assert-RecordedProcessIdentity `
            $process $root.ProcessId $root.RecordedStart $root.ProcessName `
            "$ExpectedName $($root.Role)"
        $snapshotById[[int]$root.ProcessId] = [pscustomobject]@{
            ProcessId = [int]$root.ProcessId
            ParentProcessId = 0
            Depth = 0
            Role = [string]$root.Role
            ProcessName = [string]$root.ProcessName
            RecordedStart = [string]$root.RecordedStart
            StartedAtUtc = $process.StartTime.ToUniversalTime()
        }
    }
    if ($snapshotById.Count -eq 0) { return @() }

    $processTable = @(
        Get-CimInstance -ClassName Win32_Process -ErrorAction Stop |
            Select-Object ProcessId, ParentProcessId, CreationDate
    )
    if ($processTable.Count -eq 0) {
        throw "$ExpectedName process-tree metadata is unavailable; no process was stopped."
    }

    # CIM binding prevents a recycled root from lending ownership to inferred helpers.
    foreach ($rootSnapshot in @($snapshotById.Values)) {
        $processId = [int]$rootSnapshot.ProcessId
        $rootEntries = @(
            $processTable | Where-Object { [int]$_.ProcessId -eq $processId }
        )
        if ($rootEntries.Count -ne 1) {
            throw "$ExpectedName $($rootSnapshot.Role) PID $processId has ambiguous process-tree metadata; no process was stopped."
        }
        $cimStartedAtUtc = Get-CimProcessStartUtc `
            $rootEntries[0].CreationDate "$ExpectedName $($rootSnapshot.Role) PID $processId"
        if ([Math]::Abs(($rootSnapshot.StartedAtUtc - $cimStartedAtUtc).TotalSeconds) -ge 1.0) {
            throw "$ExpectedName $($rootSnapshot.Role) PID $processId was recycled during process-tree capture; no process was stopped."
        }
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($null -eq $process) {
            throw "$ExpectedName $($rootSnapshot.Role) PID $processId exited during process-tree capture; no process was stopped."
        }
        Assert-ProcessMatchesSnapshot `
            $process $rootSnapshot "$ExpectedName $($rootSnapshot.Role) process-tree capture"
    }

    $knownIds = [Collections.Generic.HashSet[int]]::new()
    foreach ($processId in $snapshotById.Keys) { [void]$knownIds.Add([int]$processId) }
    $parentById = @{}
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
                $entry.CreationDate "$ExpectedName descendant PID $processId"
            if ([Math]::Abs(($startedAtUtc - $cimStartedAtUtc).TotalSeconds) -ge 1.0) {
                throw "$ExpectedName descendant PID $processId was recycled; no process was stopped."
            }
            $parentSnapshot = $snapshotById[$parentId]
            if (($startedAtUtc - $parentSnapshot.StartedAtUtc).TotalSeconds -lt -1.0) {
                # Windows retains ParentProcessId after exit; an older process cannot
                # belong to this verified parent after its PID has been reused.
                continue
            }
            $snapshotById[$processId] = [pscustomobject]@{
                ProcessId = $processId
                ParentProcessId = $parentId
                Depth = 0
                Role = "helper"
                ProcessName = [string]$process.ProcessName
                RecordedStart = $startedAtUtc.ToString("O")
                StartedAtUtc = $startedAtUtc
            }
            $parentById[$processId] = $parentId
            [void]$knownIds.Add($processId)
            $changed = $true
        }
    }

    # Record verified links among the roots too, including launcher -> cmd -> listener chains.
    foreach ($entry in $processTable) {
        $processId = [int]$entry.ProcessId
        $parentId = [int]$entry.ParentProcessId
        if (
            $processId -eq $launcherId -or
            -not $knownIds.Contains($processId) -or
            -not $knownIds.Contains($parentId)
        ) {
            continue
        }
        $snapshot = $snapshotById[$processId]
        $cimStartedAtUtc = Get-CimProcessStartUtc `
            $entry.CreationDate "$ExpectedName process PID $processId"
        if (
            [Math]::Abs(($snapshot.StartedAtUtc - $cimStartedAtUtc).TotalSeconds) -ge 1.0 -or
            ($snapshot.StartedAtUtc - $snapshotById[$parentId].StartedAtUtc).TotalSeconds -lt -1.0
        ) {
            throw "$ExpectedName process-tree relationship is not trustworthy; no process was stopped."
        }
        $snapshot.ParentProcessId = $parentId
        $parentById[$processId] = $parentId
    }

    foreach ($snapshot in @($snapshotById.Values)) {
        $processId = [int]$snapshot.ProcessId
        if ($processId -eq $launcherId) {
            $snapshot.Depth = 0
            continue
        }

        $currentId = $processId
        $edges = 0
        $reachedLauncher = $false
        $visited = [Collections.Generic.HashSet[int]]::new()
        while ($parentById.ContainsKey($currentId)) {
            if (-not $visited.Add($currentId)) {
                throw "$ExpectedName process tree contains a cycle; no process was stopped."
            }
            $currentId = [int]$parentById[$currentId]
            $edges += 1
            if ($currentId -eq $launcherId) {
                $snapshot.Depth = $edges
                $reachedLauncher = $true
                break
            }
        }
        if ($reachedLauncher) { continue }

        if ($processId -eq $listenerId) {
            $snapshot.Depth = 1
            continue
        }
        $currentId = $processId
        $edges = 0
        $reachedListener = $false
        $visited.Clear()
        while ($parentById.ContainsKey($currentId)) {
            if (-not $visited.Add($currentId)) {
                throw "$ExpectedName process tree contains a cycle; no process was stopped."
            }
            $currentId = [int]$parentById[$currentId]
            $edges += 1
            if ($currentId -eq $listenerId) {
                $snapshot.Depth = $edges + 1
                $reachedListener = $true
                break
            }
        }
        if (-not $reachedListener) {
            throw "$ExpectedName helper PID $processId is not rooted in trusted state; no process was stopped."
        }
    }
    return @($snapshotById.Values)
}

function Stop-OwnedService(
    [object]$Record,
    [string]$ExpectedName,
    [string]$ExpectedScript,
    [string]$ListenerMarker,
    [string]$ExpectedListenerProcessName
) {
    if ($null -eq $Record) { return }
    # Revalidate immediately before collecting and terminating the exact recorded processes.
    Assert-ServiceOwnership `
        $Record $ExpectedName $ExpectedScript $ListenerMarker $ExpectedListenerProcessName
    $snapshots = @(
        Get-OwnedProcessSnapshot $Record $ExpectedListenerProcessName $ExpectedName |
            Sort-Object Depth -Descending
    )
    # Roots must still be exact before the first inferred helper is terminated.
    Assert-OwnedRootSnapshotsCurrent $snapshots $ExpectedName
    foreach ($snapshot in $snapshots) {
        $processId = [int]$snapshot.ProcessId
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($null -eq $process) { continue }
        Assert-ProcessMatchesSnapshot `
            $process $snapshot "$ExpectedName $($snapshot.Role)"
        Stop-Process -InputObject $process -Force -ErrorAction SilentlyContinue
        if (-not $process.WaitForExit($StopTimeoutSeconds * 1000)) {
            $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
            if ($null -ne $process) {
                Assert-ProcessMatchesSnapshot `
                    $process $snapshot "$ExpectedName $($snapshot.Role) taskkill fallback"
                $previousErrorActionPreference = $ErrorActionPreference
                try {
                    # taskkill can report a non-zero race when the exact process
                    # exits after revalidation. Final snapshot checks remain authoritative.
                    $ErrorActionPreference = "Continue"
                    & $taskkill /PID $processId /F 2>$null | Out-Null
                }
                finally { $ErrorActionPreference = $previousErrorActionPreference }
                $process.WaitForExit(5000) | Out-Null
            }
        }
    }
    foreach ($snapshot in $snapshots) {
        $processId = [int]$snapshot.ProcessId
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($null -eq $process) { continue }
        if (Test-RecordedStart $process $snapshot.RecordedStart) {
            throw "$ExpectedName owned PID $processId remained alive; trusted state was retained."
        }
        Write-Warning "$ExpectedName PID $processId was recycled after its owned process exited; the new process was not touched."
    }
    Write-Host "Stopped owned $ExpectedName processes."
}

function Remove-TrustedWorkerMetadata([object]$Record) {
    if ($null -eq $Record) { return }
    # Revalidate both trusted records immediately before deleting the exact metadata file.
    Assert-WorkerOwnership $Record
    Assert-WorkerMetadataPathSafe
    if (Test-Path -LiteralPath $workerMetadataPath -PathType Leaf) {
        Remove-Item -LiteralPath $workerMetadataPath -Force
    }
}

function Stop-OwnedWorker([object]$Record) {
    if ($null -eq $Record) { return }
    Assert-WorkerOwnership $Record
    $processId = [int](Get-PropertyValue $Record "pid")
    $recordedStart = [string](Get-PropertyValue $Record "process_started_at_utc")
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        Write-Warning "Trusted MegaLoc PID is already absent; metadata is retained until all demo ports are closed."
        return
    }

    try {
        $body = @{
            schema_version = "atlaslens-worker-v1"
            request_id = "stop-$([Guid]::NewGuid().ToString('N'))"
            parameters = @{}
        } | ConvertTo-Json -Compress
        Invoke-RestMethod -Uri "http://127.0.0.1:$workerPort/v1/unload" -Method Post `
            -TimeoutSec $StopTimeoutSeconds -ContentType "application/json" `
            -Body $body | Out-Null
    }
    catch { Write-Warning "MegaLoc did not acknowledge unload; stopping the trusted process." }

    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($null -ne $process) {
        # Revalidate after graceful unload and immediately before termination.
        Assert-WorkerOwnership $Record
        Assert-RecordedProcessIdentity $process $processId $recordedStart "python" "MegaLoc worker"
        Stop-Process -InputObject $process -ErrorAction SilentlyContinue
        if (-not $process.WaitForExit($StopTimeoutSeconds * 1000)) {
            $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
            if ($null -ne $process) {
                Assert-RecordedProcessIdentity $process $processId $recordedStart "python" "MegaLoc worker"
                Stop-Process -InputObject $process -Force -ErrorAction SilentlyContinue
                if (-not $process.WaitForExit(5000)) {
                    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
                    if ($null -ne $process) {
                        Assert-RecordedProcessIdentity `
                            $process $processId $recordedStart "python" `
                            "MegaLoc worker taskkill fallback"
                        $previousErrorActionPreference = $ErrorActionPreference
                        try {
                            # A just-exited exact PID may make taskkill return non-zero.
                            # The post-stop identity check below decides success.
                            $ErrorActionPreference = "Continue"
                            & $taskkill /PID $processId /F 2>$null | Out-Null
                        }
                        finally { $ErrorActionPreference = $previousErrorActionPreference }
                        $process.WaitForExit(5000) | Out-Null
                    }
                }
            }
        }
    }

    $remaining = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($null -ne $remaining -and (Test-RecordedStart $remaining $recordedStart)) {
        throw "Owned MegaLoc process remained alive; trusted state was retained."
    }
    if ($null -ne $remaining) {
        Write-Warning "MegaLoc PID $processId was recycled after its owned process exited; the new process was not touched."
    }
    Write-Host "Stopped owned MegaLoc worker PID $processId."
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

Assert-InvestorDemoRuntimePath
if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
    $listeners = @(
        foreach ($port in @($workerPort, 8000, 5173)) {
            Get-ListeningConnections $port
        }
    )
    if ($listeners.Count -gt 0) {
        Write-Warning "No trusted investor-demo state exists; existing listeners were not touched."
    }
    else { Write-Host "Investor demo is not recorded as running." }
    return
}

Assert-InvestorDemoRuntimeTreeSafe
Assert-InvestorDemoStateFileSafe
try { $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json }
catch { throw "Investor demo state is invalid; no process was stopped." }
if (
    $state.schema_version -ne "atlaslens-investor-demo-process-v1" -or
    -not [string]::Equals([string]$state.repository_root, $repoRoot, [StringComparison]::OrdinalIgnoreCase) -or
    $state.status -notin @("starting", "ready")
) {
    throw "Investor demo state identity is invalid; no process was stopped."
}

# Validate every live target before the first termination action.
Assert-ServiceOwnership $state.web "web" $webScript "vite" "node"
Assert-ServiceOwnership $state.api "api" $apiScript "atlaslens_api.main:app" "python"
Assert-WorkerOwnership $state.worker

Stop-OwnedService $state.web "web" $webScript "vite" "node"
Stop-OwnedService $state.api "api" $apiScript "atlaslens_api.main:app" "python"
Stop-OwnedWorker $state.worker
Assert-NoInvestorDemoListeners $state
Remove-TrustedWorkerMetadata $state.worker
Remove-InvestorDemoRuntime
Write-Host "AtlasLens investor demo stopped. Private demo database, temporary files, logs, and PID state were deleted and are not recoverable."
