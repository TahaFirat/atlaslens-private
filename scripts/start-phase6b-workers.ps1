[CmdletBinding()]
param(
    [ValidateSet("osv5m", "plonk", "paddleocr")]
    [string[]]$Providers = @("osv5m", "plonk", "paddleocr"),
    [ValidateRange(5, 300)]
    [int]$StartupTimeoutSeconds = 90,
    [AllowEmptyString()][string]$ApiPython = "",
    [switch]$SkipInferenceVerification
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runRoot = Join-Path $repoRoot ".local\run\phase6b-workers"
$logRoot = Join-Path $runRoot "logs"
$protocolVersion = "atlaslens-worker-v1"
$metadataVersion = "atlaslens-worker-process-v1"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

function Get-WorkerSpec([string]$Name) {
    switch ($Name) {
        "osv5m" {
            return [pscustomobject]@{
                Name = "osv5m"
                Port = 8791
                PythonPath = Join-Path $repoRoot ".local\workers\osv5m\.venv\Scripts\python.exe"
                WorkerPath = Join-Path $repoRoot "services\model-workers\osv5m\worker.py"
                ProviderRevision = "4e6075387ecde4255410785ffb83830c9aa099f6"
                ModelRevisions = @("71548b90ac4a1aa7c37839841f411a06da82b1a6")
                Arguments = @("--host", "127.0.0.1", "--port", "8791")
                Environment = @{}
            }
        }
        "plonk" {
            return [pscustomobject]@{
                Name = "plonk"
                Port = 8792
                PythonPath = Join-Path $repoRoot ".local\workers\plonk\.venv\Scripts\python.exe"
                WorkerPath = Join-Path $repoRoot "services\model-workers\plonk\worker.py"
                ProviderRevision = "76d46410910c9dfec9e19ed371450ebc7051cdf3"
                ModelRevisions = @(
                    "scene-routed",
                    "e23229f4dd91d52560e8827f5bb2c68257fa162f",
                    "4f358d09938a89ed239a847777729e95c5d187bc",
                    "8da6edcbdd01ff04a61f9d06e2de23ea300d1a35"
                )
                Arguments = @("--host", "127.0.0.1", "--port", "8792", "--num-steps", "8")
                Environment = @{}
            }
        }
        "paddleocr" {
            return [pscustomobject]@{
                Name = "paddleocr"
                Port = 8793
                PythonPath = Join-Path $repoRoot ".local\workers\paddleocr\.venv\Scripts\python.exe"
                WorkerPath = Join-Path $repoRoot "services\model-workers\paddleocr\worker.py"
                ProviderRevision = "3.7.0"
                ModelRevisions = @("PP-OCRv5_server_det+latin_PP-OCRv5_mobile_rec")
                Arguments = @()
                Environment = @{
                    PADDLEOCR_WORKER_HOST = "127.0.0.1"
                    PADDLEOCR_WORKER_PORT = "8793"
                }
            }
        }
        default { throw "Unsupported worker '$Name'." }
    }
}

function Get-PropertyValue([object]$Value, [string]$Name) {
    if ($null -eq $Value) { return $null }
    $property = $Value.PSObject.Properties[$Name]
    if ($null -eq $property) { return $null }
    return $property.Value
}

function Test-HealthIdentity([object]$Health, [object]$Spec) {
    $revision = [string](Get-PropertyValue $Health "model_revision")
    return (
        (Get-PropertyValue $Health "schema_version") -eq $protocolVersion -and
        (Get-PropertyValue $Health "provider") -eq $Spec.Name -and
        (Get-PropertyValue $Health "provider_revision") -eq $Spec.ProviderRevision -and
        (Get-PropertyValue $Health "process_running") -eq $true -and
        $Spec.ModelRevisions -contains $revision
    )
}

function Get-WorkerHealth([object]$Spec) {
    try {
        return Invoke-RestMethod -Uri "http://127.0.0.1:$($Spec.Port)/health" -Method Get -TimeoutSec 2 -Headers @{ Accept = "application/json" }
    }
    catch { return $null }
}

function Get-PortOwner([int]$Port) {
    $listeners = @(
        Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
            Where-Object { $_.LocalAddress -eq "127.0.0.1" }
    )
    if ($listeners.Count -eq 0) { return $null }
    $owners = @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)
    if ($owners.Count -ne 1) { throw "Port $Port has ambiguous loopback listeners." }
    return [int]$owners[0]
}

function Get-CommandInfo([int]$ProcessId) {
    return Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
}

function Test-CommandIdentity([object]$CommandInfo, [object]$Spec) {
    if ($null -eq $CommandInfo -or [string]::IsNullOrWhiteSpace($CommandInfo.ExecutablePath) -or [string]::IsNullOrWhiteSpace($CommandInfo.CommandLine)) {
        return $false
    }
    try { $actualPython = [IO.Path]::GetFullPath([string]$CommandInfo.ExecutablePath) }
    catch { return $false }
    $allowedPython = @([IO.Path]::GetFullPath([string]$Spec.PythonPath))
    try {
        $basePython = (& $Spec.PythonPath -c "import sys; print(sys._base_executable)" 2>$null).Trim()
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($basePython)) {
            $allowedPython += [IO.Path]::GetFullPath($basePython)
        }
    }
    catch { return $false }
    if (-not ($allowedPython | Where-Object { [string]::Equals($actualPython, $_, [StringComparison]::OrdinalIgnoreCase) })) {
        return $false
    }
    $commandLine = ([string]$CommandInfo.CommandLine).Replace("/", "\")
    $workerPath = ([string]$Spec.WorkerPath).Replace("/", "\")
    return $commandLine.IndexOf($workerPath, [StringComparison]::OrdinalIgnoreCase) -ge 0
}

function Get-MetadataPath([object]$Spec) {
    return Join-Path $runRoot "$($Spec.Name).json"
}

function Read-PidRecord([object]$Spec) {
    $path = Get-MetadataPath $Spec
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
    try { return Get-Content -Raw -LiteralPath $path | ConvertFrom-Json }
    catch { throw "PID metadata for $($Spec.Name) is invalid; refusing to overwrite it." }
}

function Remove-PidRecord([object]$Spec) {
    $path = Get-MetadataPath $Spec
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        Remove-Item -LiteralPath $path -Force
    }
}

function Write-PidRecord([object]$Spec, [System.Diagnostics.Process]$Process, [string]$StdoutPath, [string]$StderrPath) {
    $path = Get-MetadataPath $Spec
    $temporary = "$path.$PID.tmp"
    $record = [ordered]@{
        schema_version = $metadataVersion
        provider = $Spec.Name
        pid = $Process.Id
        port = $Spec.Port
        python_path = $Spec.PythonPath
        worker_path = $Spec.WorkerPath
        provider_revision = $Spec.ProviderRevision
        allowed_model_revisions = $Spec.ModelRevisions
        process_started_at_utc = $Process.StartTime.ToUniversalTime().ToString("O")
        stdout_path = $StdoutPath
        stderr_path = $StderrPath
    }
    $record | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $path -Force
}

function Test-RecordProcess([object]$Record, [System.Diagnostics.Process]$Process, [object]$Spec) {
    if ($null -eq $Record -or (Get-PropertyValue $Record "schema_version") -ne $metadataVersion) { return $false }
    if ((Get-PropertyValue $Record "provider") -ne $Spec.Name -or [int](Get-PropertyValue $Record "pid") -ne $Process.Id) { return $false }
    try {
        $recorded = [DateTimeOffset]::Parse([string](Get-PropertyValue $Record "process_started_at_utc")).UtcDateTime
        $actual = $Process.StartTime.ToUniversalTime()
        return [Math]::Abs(($recorded - $actual).TotalSeconds) -lt 1.0
    }
    catch { return $false }
}

function Quote-ProcessArgument([string]$Value) {
    if ($Value.Contains('"')) { throw "Worker path contains an unsupported quote." }
    return '"' + $Value + '"'
}

function Start-WithEnvironment([object]$Spec, [string]$StdoutPath, [string]$StderrPath) {
    $values = @{
        PYTHONUNBUFFERED = "1"
        HF_HUB_OFFLINE = "1"
        TRANSFORMERS_OFFLINE = "1"
        HF_HUB_DISABLE_TELEMETRY = "1"
    }
    foreach ($entry in $Spec.Environment.GetEnumerator()) { $values[$entry.Key] = $entry.Value }
    # Some managed/sandboxed Windows hosts expose both `Path` and `PATH` in the
    # process block. Start-Process treats them case-insensitively and otherwise
    # fails before launch, so temporarily retain one exact key for inheritance.
    $pathEntries = @()
    $environmentBlock = [Environment]::GetEnvironmentVariables()
    foreach ($key in $environmentBlock.Keys) {
        if ([string]::Equals([string]$key, "Path", [StringComparison]::OrdinalIgnoreCase)) {
            $pathEntries += [pscustomobject]@{ Key = [string]$key; Value = [string]$environmentBlock[$key] }
        }
    }
    $removedPathEntries = @()
    if ($pathEntries.Count -gt 1) {
        $canonical = @($pathEntries | Where-Object { $_.Key -ceq "Path" } | Select-Object -First 1)
        $canonicalKey = if ($canonical.Count -eq 1) { $canonical[0].Key } else { $pathEntries[0].Key }
        foreach ($entry in $pathEntries) {
            if ($entry.Key -cne $canonicalKey) {
                [Environment]::SetEnvironmentVariable($entry.Key, $null, [EnvironmentVariableTarget]::Process)
                $removedPathEntries += $entry
            }
        }
    }
    $previous = @{}
    foreach ($entry in $values.GetEnumerator()) {
        $item = Get-Item -LiteralPath "Env:$($entry.Key)" -ErrorAction SilentlyContinue
        $previous[$entry.Key] = if ($null -eq $item) { $null } else { $item.Value }
        Set-Item -LiteralPath "Env:$($entry.Key)" -Value $entry.Value
    }
    try {
        $arguments = @((Quote-ProcessArgument $Spec.WorkerPath)) + $Spec.Arguments
        return Start-Process -FilePath $Spec.PythonPath -ArgumentList $arguments -WorkingDirectory $repoRoot -PassThru -WindowStyle Hidden -RedirectStandardOutput $StdoutPath -RedirectStandardError $StderrPath
    }
    finally {
        foreach ($entry in $previous.GetEnumerator()) {
            if ($null -eq $entry.Value) { Remove-Item -LiteralPath "Env:$($entry.Key)" -ErrorAction SilentlyContinue }
            else { Set-Item -LiteralPath "Env:$($entry.Key)" -Value $entry.Value }
        }
        foreach ($entry in $removedPathEntries) {
            [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, [EnvironmentVariableTarget]::Process)
        }
    }
}

function Wait-WorkerReady([object]$Spec, [System.Diagnostics.Process]$Process) {
    $deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
    do {
        $Process.Refresh()
        $owner = Get-PortOwner $Spec.Port
        if ($null -ne $owner) {
            $command = Get-CommandInfo $owner
            if (-not (Test-CommandIdentity $command $Spec)) {
                throw "Port $($Spec.Port) was claimed by foreign PID $owner during startup."
            }
            $health = Get-WorkerHealth $Spec
            if (Test-HealthIdentity $health $Spec) {
                return [pscustomobject]@{
                    Health = $health
                    Process = Get-Process -Id $owner -ErrorAction Stop
                }
            }
        }
        if ($Process.HasExited) { throw "$($Spec.Name) worker exited before readiness with code $($Process.ExitCode)." }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)
    throw "$($Spec.Name) worker readiness timed out on 127.0.0.1:$($Spec.Port)."
}

$started = @()
try {
    foreach ($name in $Providers) {
        $spec = Get-WorkerSpec $name
        foreach ($required in @($spec.PythonPath, $spec.WorkerPath)) {
            if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Required $name worker file is missing." }
        }
        $owner = Get-PortOwner $spec.Port
        if ($null -ne $owner) {
            $health = Get-WorkerHealth $spec
            $command = Get-CommandInfo $owner
            if (-not (Test-HealthIdentity $health $spec) -or -not (Test-CommandIdentity $command $spec)) {
                throw "Port $($spec.Port) is occupied by foreign or identity-mismatched PID $owner; it was not touched."
            }
            $process = Get-Process -Id $owner -ErrorAction Stop
            Write-PidRecord $spec $process "" ""
            Write-Host "Reusing $name worker: PID $owner, 127.0.0.1:$($spec.Port)."
            continue
        }

        $oldRecord = Read-PidRecord $spec
        if ($null -ne $oldRecord) {
            $oldProcessId = [int](Get-PropertyValue $oldRecord "pid")
            $oldProcess = Get-Process -Id $oldProcessId -ErrorAction SilentlyContinue
            if ($null -eq $oldProcess) {
                Write-Warning "Removed stale $name PID metadata for exited PID $oldProcessId."
                Remove-PidRecord $spec
            }
            elseif (Test-RecordProcess $oldRecord $oldProcess $spec -and (Test-CommandIdentity (Get-CommandInfo $oldProcessId) $spec)) {
                throw "Stale $name worker PID $oldProcessId is alive without its expected listener; run stop-phase6b-workers.ps1."
            }
            else {
                throw "Stale $name metadata points at a different live process; it was not touched."
            }
        }

        $stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ")
        $token = [Guid]::NewGuid().ToString("N").Substring(0, 8)
        $stdoutPath = Join-Path $logRoot "$name-$stamp-$token.out.log"
        $stderrPath = Join-Path $logRoot "$name-$stamp-$token.err.log"
        $process = Start-WithEnvironment $spec $stdoutPath $stderrPath
        $startedEntry = [pscustomobject]@{ Spec = $spec; Process = $process; Launcher = $process }
        $started += $startedEntry
        $ready = Wait-WorkerReady $spec $process
        $startedEntry.Process = $ready.Process
        Write-PidRecord $spec $ready.Process $stdoutPath $stderrPath
        Write-Host "Started $name worker: PID $($ready.Process.Id), 127.0.0.1:$($spec.Port)."
    }
    if (-not $SkipInferenceVerification) {
        $apiPython = if ([string]::IsNullOrWhiteSpace($ApiPython)) {
            Join-Path $repoRoot "services\api\.venv\Scripts\python.exe"
        }
        elseif (-not [IO.Path]::IsPathRooted($ApiPython) -or -not (Test-Path -LiteralPath $ApiPython -PathType Leaf)) {
            throw "ApiPython must be a full path to an existing python.exe."
        }
        else {
            (Resolve-Path -LiteralPath $ApiPython).Path
        }
        $verifier = Join-Path $PSScriptRoot "verify_phase6b_workers.py"
        foreach ($required in @($apiPython, $verifier)) {
            if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
                throw "Required real-inference verifier file is missing."
            }
        }
        $verifyArguments = @($verifier, "--project-root", $repoRoot, "--providers") + $Providers
        & $apiPython @verifyArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Phase 6B worker real-inference verification failed with code $LASTEXITCODE."
        }
        Write-Host "Selected Phase 6B workers passed real load and inference verification."
    }
}
catch {
    foreach ($entry in $started) {
        foreach ($candidate in @($entry.Process, $entry.Launcher)) {
            $candidate.Refresh()
            if (-not $candidate.HasExited) {
                Stop-Process -Id $candidate.Id -Force -ErrorAction SilentlyContinue
                $candidate.WaitForExit(5000) | Out-Null
            }
        }
        Remove-PidRecord $entry.Spec
    }
    throw
}
