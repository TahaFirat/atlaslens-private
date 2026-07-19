[CmdletBinding()]
param(
    [ValidateRange(1, 60)]
    [int]$GracefulTimeoutSeconds = 10,
    [switch]$IncludePhase6B
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$metadataPath = Join-Path $repoRoot ".local\run\phase6c-workers\megaloc.json"
$port = 8794

if (-not (Test-Path -LiteralPath $metadataPath -PathType Leaf)) {
    $listener = @(Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalAddress -eq "127.0.0.1" })
    if ($listener.Count) {
        Write-Warning "MegaLoc has an untrusted listener on port $port; it was not touched."
    }
    else { Write-Host "MegaLoc worker is not recorded as running." }
}
else {
    $record = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    if ($record.schema_version -ne "atlaslens-worker-process-v1" -or $record.provider -ne "megaloc") {
        throw "MegaLoc PID metadata is invalid; no process was stopped."
    }
    $process = Get-Process -Id ([int]$record.pid) -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        Remove-Item -LiteralPath $metadataPath -Force
        Write-Warning "Removed stale MegaLoc PID metadata."
    }
    else {
        $started = [DateTimeOffset]::Parse([string]$record.process_started_at_utc).UtcDateTime
        if ([Math]::Abs(($started - $process.StartTime.ToUniversalTime()).TotalSeconds) -ge 1) {
            throw "MegaLoc PID was recycled; no process was stopped."
        }
        $owner = @(Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue |
            Where-Object { $_.LocalAddress -eq "127.0.0.1" } |
            Select-Object -ExpandProperty OwningProcess -Unique)
        if ($owner.Count -ne 1 -or [int]$owner[0] -ne $process.Id) {
            throw "MegaLoc listener ownership changed; no process was stopped."
        }
        try {
            $body = @{
                schema_version = "atlaslens-worker-v1"
                request_id = "stop-$([Guid]::NewGuid().ToString('N'))"
                parameters = @{}
            } | ConvertTo-Json -Compress
            Invoke-RestMethod -Uri "http://127.0.0.1:$port/v1/unload" -Method Post `
                -TimeoutSec $GracefulTimeoutSeconds -ContentType "application/json" `
                -Body $body | Out-Null
        }
        catch { Write-Warning "MegaLoc did not acknowledge unload; stopping the trusted process." }
        Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
        if (-not $process.WaitForExit($GracefulTimeoutSeconds * 1000)) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            if (-not $process.WaitForExit(5000)) { throw "MegaLoc worker did not stop." }
        }
        Remove-Item -LiteralPath $metadataPath -Force
        Write-Host "Stopped MegaLoc worker PID $($process.Id), port $port."
    }
}

if ($IncludePhase6B) {
    & (Join-Path $PSScriptRoot "stop-phase6b-workers.ps1")
}
