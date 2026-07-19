[CmdletBinding()]
param(
    [ValidateSet("osv5m", "plonk", "paddleocr")]
    [string[]]$Providers = @("osv5m", "plonk", "paddleocr"),
    [ValidateRange(1, 60)]
    [int]$GracefulTimeoutSeconds = 10,
    [ValidateRange(1, 30)]
    [int]$ForceTimeoutSeconds = 5
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runRoot = Join-Path $repoRoot ".local\run\phase6b-workers"
$protocolVersion = "atlaslens-worker-v1"
$metadataVersion = "atlaslens-worker-process-v1"

function Get-WorkerSpec([string]$Name) {
    switch ($Name) {
        "osv5m" {
            return [pscustomobject]@{
                Name = "osv5m"; Port = 8791
                PythonPath = Join-Path $repoRoot ".local\workers\osv5m\.venv\Scripts\python.exe"
                WorkerPath = Join-Path $repoRoot "services\model-workers\osv5m\worker.py"
                ProviderRevision = "4e6075387ecde4255410785ffb83830c9aa099f6"
                ModelRevisions = @("71548b90ac4a1aa7c37839841f411a06da82b1a6")
            }
        }
        "plonk" {
            return [pscustomobject]@{
                Name = "plonk"; Port = 8792
                PythonPath = Join-Path $repoRoot ".local\workers\plonk\.venv\Scripts\python.exe"
                WorkerPath = Join-Path $repoRoot "services\model-workers\plonk\worker.py"
                ProviderRevision = "76d46410910c9dfec9e19ed371450ebc7051cdf3"
                ModelRevisions = @(
                    "scene-routed",
                    "e23229f4dd91d52560e8827f5bb2c68257fa162f",
                    "4f358d09938a89ed239a847777729e95c5d187bc",
                    "8da6edcbdd01ff04a61f9d06e2de23ea300d1a35"
                )
            }
        }
        "paddleocr" {
            return [pscustomobject]@{
                Name = "paddleocr"; Port = 8793
                PythonPath = Join-Path $repoRoot ".local\workers\paddleocr\.venv\Scripts\python.exe"
                WorkerPath = Join-Path $repoRoot "services\model-workers\paddleocr\worker.py"
                ProviderRevision = "3.7.0"
                ModelRevisions = @("PP-OCRv5_server_det+latin_PP-OCRv5_mobile_rec")
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
    if ($null -eq $Health) { return $false }
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
    if (-not ($allowedPython | Where-Object { [string]::Equals($actualPython, $_, [StringComparison]::OrdinalIgnoreCase) })) { return $false }
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
    catch { throw "PID metadata for $($Spec.Name) is invalid; no process was stopped." }
}

function Test-RecordIdentity([object]$Record, [System.Diagnostics.Process]$Process, [object]$Spec) {
    if ((Get-PropertyValue $Record "schema_version") -ne $metadataVersion) { return $false }
    if ((Get-PropertyValue $Record "provider") -ne $Spec.Name -or [int](Get-PropertyValue $Record "pid") -ne $Process.Id) { return $false }
    if ([int](Get-PropertyValue $Record "port") -ne $Spec.Port) { return $false }
    if (-not [string]::Equals([string](Get-PropertyValue $Record "python_path"), $Spec.PythonPath, [StringComparison]::OrdinalIgnoreCase)) { return $false }
    if (-not [string]::Equals([string](Get-PropertyValue $Record "worker_path"), $Spec.WorkerPath, [StringComparison]::OrdinalIgnoreCase)) { return $false }
    try {
        $recorded = [DateTimeOffset]::Parse([string](Get-PropertyValue $Record "process_started_at_utc")).UtcDateTime
        $actual = $Process.StartTime.ToUniversalTime()
        return [Math]::Abs(($recorded - $actual).TotalSeconds) -lt 1.0
    }
    catch { return $false }
}

function Remove-PidRecord([object]$Spec) {
    $path = Get-MetadataPath $Spec
    if (Test-Path -LiteralPath $path -PathType Leaf) { Remove-Item -LiteralPath $path -Force }
}

function Invoke-WorkerUnload([object]$Spec) {
    $body = @{
        schema_version = $protocolVersion
        request_id = "stop-$([Guid]::NewGuid().ToString('N'))"
        parameters = @{}
    } | ConvertTo-Json -Compress
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:$($Spec.Port)/v1/unload" -Method Post -TimeoutSec $GracefulTimeoutSeconds -ContentType "application/json" -Body $body
        return (Get-PropertyValue $response "ok") -eq $true
    }
    catch { return $false }
}

foreach ($name in $Providers) {
    $spec = Get-WorkerSpec $name
    $record = Read-PidRecord $spec
    if ($null -eq $record) {
        $owner = Get-PortOwner $spec.Port
        if ($null -ne $owner) { Write-Warning "$name has a listener at 127.0.0.1:$($spec.Port) but no trusted PID metadata; PID $owner was not touched." }
        else { Write-Host "$name worker is not recorded as running." }
        continue
    }

    $processId = [int](Get-PropertyValue $record "pid")
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        Write-Warning "Removed stale $name PID metadata for exited PID $processId."
        Remove-PidRecord $spec
        continue
    }
    $command = Get-CommandInfo $processId
    if (-not (Test-RecordIdentity $record $process $spec) -or -not (Test-CommandIdentity $command $spec)) {
        throw "$name PID metadata does not match PID $processId; no process was stopped."
    }

    $owner = Get-PortOwner $spec.Port
    if ($null -ne $owner -and $owner -ne $processId) {
        throw "$name port $($spec.Port) belongs to foreign PID $owner; no process was stopped."
    }
    if ($null -ne $owner) {
        $health = Get-WorkerHealth $spec
        if ($null -ne $health -and -not (Test-HealthIdentity $health $spec)) {
            throw "$name health identity does not match PID $processId; no process was stopped."
        }
        if (-not (Invoke-WorkerUnload $spec)) {
            Write-Warning "$name did not acknowledge model unload; stopping the validated worker process."
        }
    }

    # Revalidate immediately before termination to avoid acting on a recycled PID.
    $process.Refresh()
    if ($process.HasExited) {
        Remove-PidRecord $spec
        Write-Host "Stopped $name worker: PID $processId."
        continue
    }
    if (-not (Test-RecordIdentity $record $process $spec) -or -not (Test-CommandIdentity (Get-CommandInfo $processId) $spec)) {
        throw "$name process identity changed before stop; no process was stopped."
    }
    Stop-Process -Id $processId -ErrorAction SilentlyContinue
    if (-not $process.WaitForExit($GracefulTimeoutSeconds * 1000)) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        if (-not $process.WaitForExit($ForceTimeoutSeconds * 1000)) {
            throw "$name worker PID $processId did not stop within the bounded timeout."
        }
    }
    Remove-PidRecord $spec
    Write-Host "Stopped $name worker: PID $processId, port $($spec.Port)."
}
