[CmdletBinding()]
param(
    [ValidateRange(10, 300)]
    [int]$StartupTimeoutSeconds = 180,
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",
    [switch]$SkipInferenceVerification,
    [switch]$IncludePhase6B
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runRoot = Join-Path $repoRoot ".local\run\phase6c-workers"
$logRoot = Join-Path $runRoot "logs"
$metadataPath = Join-Path $runRoot "megaloc.json"
$python = Join-Path $repoRoot ".local\workers\megaloc\.venv\Scripts\python.exe"
$worker = Join-Path $repoRoot "services\model-workers\megaloc\worker.py"
$port = 8794
$sourceRevision = "1af071c68fc3ab6c6018c5c868391763516e50f7"
$modelRevision = "7cb9f7970d366fdf059963d04d372e503e8e9df9"

function Get-PortOwner {
    $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalAddress -eq "127.0.0.1" })
    if ($listeners.Count -eq 0) { return $null }
    $owners = @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)
    if ($owners.Count -ne 1) { throw "Port $port has ambiguous loopback listeners." }
    return [int]$owners[0]
}

function Get-Health {
    try {
        return Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 5 -Headers @{ Accept = "application/json" }
    }
    catch { return $null }
}

function Test-Health([object]$health) {
    return (
        $null -ne $health -and
        $health.schema_version -eq "atlaslens-worker-v1" -and
        $health.provider -eq "megaloc" -and
        $health.provider_revision -eq $sourceRevision -and
        $health.model_revision -eq $modelRevision -and
        $health.process_running -eq $true
    )
}

if ($IncludePhase6B) {
    & (Join-Path $PSScriptRoot "start-phase6b-workers.ps1")
    if ($LASTEXITCODE -ne 0) { throw "Phase 6B worker startup failed." }
}
foreach ($required in @($python, $worker)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required MegaLoc worker file is missing; run bootstrap_phase6c_models.ps1."
    }
}
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$owner = Get-PortOwner
if ($null -ne $owner) {
    $health = Get-Health
    if (-not (Test-Health $health)) {
        throw "Port $port is occupied by a foreign or revision-mismatched process; it was not touched."
    }
    if (-not (Test-Path -LiteralPath $metadataPath -PathType Leaf)) {
        throw "A valid worker is listening without trusted Phase 6C PID metadata; it was not adopted."
    }
    $record = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    $ownerProcess = Get-Process -Id $owner -ErrorAction SilentlyContinue
    if (
        $record.schema_version -ne "atlaslens-worker-process-v1" -or
        $record.provider -ne "megaloc" -or
        [int]$record.pid -ne $owner -or
        $record.provider_revision -ne $sourceRevision -or
        $record.model_revision -ne $modelRevision -or
        $null -eq $ownerProcess
    ) {
        throw "MegaLoc PID metadata does not match the loopback listener; it was not adopted."
    }
    $recordedStart = [DateTimeOffset]::Parse(
        [string]$record.process_started_at_utc
    ).UtcDateTime
    if ([Math]::Abs(($recordedStart - $ownerProcess.StartTime.ToUniversalTime()).TotalSeconds) -ge 1) {
        throw "MegaLoc listener PID was recycled; it was not adopted."
    }
    Write-Host "Reusing trusted MegaLoc worker PID $owner on 127.0.0.1:$port."
}
else {
    if (Test-Path -LiteralPath $metadataPath -PathType Leaf) {
        $old = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
        $oldProcess = Get-Process -Id ([int]$old.pid) -ErrorAction SilentlyContinue
        if ($null -ne $oldProcess) {
            throw "Trusted metadata points to a live process without the expected listener; stop it first."
        }
        Remove-Item -LiteralPath $metadataPath -Force
    }
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ")
    $stdout = Join-Path $logRoot "megaloc-$stamp.out.log"
    $stderr = Join-Path $logRoot "megaloc-$stamp.err.log"
    $oldOffline = $env:HF_HUB_OFFLINE
    $oldTransformers = $env:TRANSFORMERS_OFFLINE
    try {
        $env:HF_HUB_OFFLINE = "1"
        $env:TRANSFORMERS_OFFLINE = "1"
        $process = Start-Process -FilePath $python -ArgumentList @(
            ('"' + $worker + '"'), "--host", "127.0.0.1", "--port", "$port"
        ) -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    }
    finally {
        $env:HF_HUB_OFFLINE = $oldOffline
        $env:TRANSFORMERS_OFFLINE = $oldTransformers
    }
    try {
        $deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
        do {
            $process.Refresh()
            if ($process.HasExited) { throw "MegaLoc worker exited before readiness." }
            $health = Get-Health
            if (Test-Health $health) { break }
            Start-Sleep -Milliseconds 250
        } while ((Get-Date) -lt $deadline)
        if (-not (Test-Health $health)) { throw "MegaLoc worker readiness timed out." }
        $listenerOwner = Get-PortOwner
        if ($null -eq $listenerOwner) {
            throw "MegaLoc worker passed health without a loopback listener."
        }
        $listenerProcess = Get-Process -Id $listenerOwner -ErrorAction SilentlyContinue
        if ($null -eq $listenerProcess) {
            throw "MegaLoc loopback listener process disappeared before recording."
        }
        if ($listenerOwner -ne $process.Id) {
            $listenerCim = Get-CimInstance Win32_Process -Filter "ProcessId = $listenerOwner"
            if ($null -eq $listenerCim -or [int]$listenerCim.ParentProcessId -ne $process.Id) {
                throw "MegaLoc loopback listener is not the trusted launcher or its child."
            }
        }
        $record = [ordered]@{
            schema_version = "atlaslens-worker-process-v1"
            provider = "megaloc"
            pid = $listenerOwner
            launcher_pid = $process.Id
            port = $port
            process_started_at_utc = $listenerProcess.StartTime.ToUniversalTime().ToString("O")
            python_path = $python
            worker_path = $worker
            provider_revision = $sourceRevision
            model_revision = $modelRevision
        }
        $json = $record | ConvertTo-Json -Depth 3
        [IO.File]::WriteAllText($metadataPath, $json, [Text.UTF8Encoding]::new($false))
        Write-Host "Started MegaLoc worker PID $listenerOwner on 127.0.0.1:$port."
    }
    catch {
        if ($null -ne $process -and -not $process.HasExited) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
        throw
    }
}

if (-not $SkipInferenceVerification) {
    $apiPython = Join-Path $repoRoot "services\api\.venv\Scripts\python.exe"
    $oldPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = Join-Path $repoRoot "services\api\src"
        & $apiPython (Join-Path $PSScriptRoot "verify_phase6c_workers.py") `
            --port $port --device $Device --timeout-seconds $StartupTimeoutSeconds
        if ($LASTEXITCODE -ne 0) { throw "MegaLoc real worker inference verification failed." }
    }
    finally {
        $env:PYTHONPATH = $oldPythonPath
    }
    Write-Host "MegaLoc worker passed real load and inference verification."
}
