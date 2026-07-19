[CmdletBinding()]
param(
    [int]$ApiPort = $(if ($env:API_PORT) { [int]$env:API_PORT } else { 8000 }),
    [int]$WebPort = $(if ($env:VITE_DEV_PORT) { [int]$env:VITE_DEV_PORT } else { 5173 }),
    [AllowEmptyString()][string]$RuntimeRoot = $(if ($env:ATLASLENS_RUNTIME_ROOT) { $env:ATLASLENS_RUNTIME_ROOT } else { "" }),
    [switch]$IgnoreProjectEnv,
    [switch]$SmokeTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$apiScript = Join-Path $PSScriptRoot "start-api.ps1"
$webScript = Join-Path $PSScriptRoot "start-web.ps1"
$workerStartScript = Join-Path $PSScriptRoot "start-phase6b-workers.ps1"
$workerStopScript = Join-Path $PSScriptRoot "stop-phase6b-workers.ps1"
$powershell = (Get-Command "powershell.exe" -ErrorAction Stop).Source
$runtimeSupport = Join-Path $PSScriptRoot "runtime\AtlasLensRuntime.ps1"
. $runtimeSupport
$runtime = Get-AtlasLensRuntimeConfiguration -RepositoryRoot $repoRoot -RuntimeRoot $RuntimeRoot
$runtimeSnapshot = Set-AtlasLensRuntimeEnvironment -Configuration $runtime -IncludeCaches -IncludeOfflineModelGuards
try { Assert-AtlasLensApiRuntime $runtime }
finally { Restore-AtlasLensRuntimeEnvironment $runtimeSnapshot }

if ($null -eq (Get-Command "npm.cmd" -ErrorAction SilentlyContinue)) {
    throw "Required command 'npm.cmd' was not found on PATH."
}

function Assert-PortFree([int]$Port, [string]$Label) {
    $listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($listener) {
        $process = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
        $name = if ($process) { $process.ProcessName } else { "unknown" }
        throw "$Label port $Port is already owned by PID $($listener.OwningProcess) ($name). Stop that service or choose another port."
    }
}

function Assert-AtlasLensApiOrReuse([int]$Port) {
    $listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $listener) { return $false }

    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/v1/health" -TimeoutSec 2 -Headers @{ Accept = "application/json" }
        $ready = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/v1/ready" -TimeoutSec 2 -Headers @{ Accept = "application/json" }
        if ($health.status -eq "ok" -and -not [string]::IsNullOrWhiteSpace($health.version) -and $ready.status -eq "ready") {
            return $true
        }
    }
    catch {
        # A different service, or an AtlasLens API that is not ready, must not be replaced.
    }

    $process = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
    $name = if ($process) { $process.ProcessName } else { "unknown" }
    throw "API port $Port is already owned by PID $($listener.OwningProcess) ($name). Stop that service or choose another port."
}

function Assert-AtlasLensWebOrReuse([int]$Port) {
    $listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $listener) { return $false }

    try {
        $capabilities = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/v1/capabilities" -TimeoutSec 2 -Headers @{ Accept = "application/json" }
        if ($null -ne $capabilities.providers.exif -and -not [string]::IsNullOrWhiteSpace($capabilities.version)) {
            return $true
        }
    }
    catch {
        # A different service, or an AtlasLens web server without a working API proxy, must not be replaced.
    }

    $process = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
    $name = if ($process) { $process.ProcessName } else { "unknown" }
    throw "Web port $Port is already owned by PID $($listener.OwningProcess) ($name). Stop that service or choose another port."
}

function Wait-AtlasLensEndpoint(
    [string]$Url,
    [scriptblock]$Validate,
    [System.Diagnostics.Process]$Process,
    [string]$Label,
    [int]$TimeoutSeconds = 45
) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        if ($Process.HasExited) {
            throw "$Label process exited before readiness with code $($Process.ExitCode)."
        }
        try {
            $response = Invoke-RestMethod -Uri $Url -TimeoutSec 2 -Headers @{ Accept = "application/json" }
            if (& $Validate $response) { return }
        }
        catch {
            # Startup is still in progress; the final timeout reports the stable URL only.
        }
        Start-Sleep -Milliseconds 300
    } while ((Get-Date) -lt $deadline)
    throw "AtlasLens readiness check timed out at $Url."
}

function Stop-AtlasLensProcessTree([System.Diagnostics.Process]$RootProcess) {
    if ($null -eq $RootProcess) { return }

    $rootId = $RootProcess.Id
    $rootStartedAt = $RootProcess.StartTime.AddSeconds(-2)
    $processTable = @(Get-CimInstance -ClassName Win32_Process -ErrorAction SilentlyContinue |
        Select-Object ProcessId, ParentProcessId, CreationDate)
    $frontier = @($rootId)
    $descendants = [System.Collections.Generic.List[int]]::new()
    while ($frontier.Count -gt 0) {
        $parents = @($frontier)
        $frontier = @(
            $processTable |
                Where-Object {
                    [int]$_.ParentProcessId -in $parents -and
                    $null -ne $_.CreationDate -and
                    $_.CreationDate -ge $rootStartedAt
                } |
                ForEach-Object { [int]$_.ProcessId }
        )
        foreach ($processId in $frontier) { $descendants.Add($processId) }
    }

    $ordered = $descendants.ToArray()
    [array]::Reverse($ordered)
    foreach ($processId in $ordered) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }
    if (-not $RootProcess.HasExited) {
        Stop-Process -Id $rootId -Force -ErrorAction SilentlyContinue
        $RootProcess.WaitForExit()
    }
}

$reuseApi = Assert-AtlasLensApiOrReuse -Port $ApiPort
$reuseWeb = Assert-AtlasLensWebOrReuse -Port $WebPort

$startedWorkerProviders = @()
$degradedWorkerProviders = @()
if ($reuseApi) {
    Write-Host "Reusing the existing healthy AtlasLens API at http://127.0.0.1:$ApiPort"
}
else {
    foreach ($worker in @(
        [pscustomobject]@{ Name = "paddleocr"; Port = 8793 },
        [pscustomobject]@{ Name = "osv5m"; Port = 8791 },
        [pscustomobject]@{ Name = "plonk"; Port = 8792 }
    )) {
        $listenerBefore = Get-NetTCPConnection -State Listen -LocalPort $worker.Port -ErrorAction SilentlyContinue | Select-Object -First 1
        & $powershell -NoProfile -ExecutionPolicy Bypass -File $workerStartScript -Providers $worker.Name -ApiPython $runtime.ApiPython
        $workerExitCode = $LASTEXITCODE
        if ($workerExitCode -ne 0) {
            $degradedWorkerProviders += $worker.Name
            Write-Warning "Optional $($worker.Name) worker is unavailable (startup exit $workerExitCode); core API startup will continue and capabilities will report the provider state."
        }
        elseif ($null -eq $listenerBefore) {
            $startedWorkerProviders += $worker.Name
        }
    }
}

$apiProcess = $null
if (-not $reuseApi) {
    $apiArgs = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ('"' + $apiScript + '"'),
        "-ApiHost", "127.0.0.1", "-ApiPort", [string]$ApiPort
    )
    if ($runtime.ExternalRuntime) { $apiArgs += @("-RuntimeRoot", ('"' + $runtime.RuntimeRoot + '"')) }
    if ($IgnoreProjectEnv) { $apiArgs += "-IgnoreProjectEnv" }
    $apiProcess = Start-Process -FilePath $powershell -ArgumentList $apiArgs -PassThru -WindowStyle Hidden
}
$webProcess = $null

try {
    if (-not $reuseApi) {
        Wait-AtlasLensEndpoint -Url "http://127.0.0.1:$ApiPort/api/v1/health" -Process $apiProcess -Label "API" -Validate {
            param($body) $body.status -eq "ok" -and -not [string]::IsNullOrWhiteSpace($body.version)
        }
        Wait-AtlasLensEndpoint -Url "http://127.0.0.1:$ApiPort/api/v1/ready" -Process $apiProcess -Label "API" -Validate {
            param($body) $body.status -eq "ready"
        }
    }

    $apiTarget = "http://127.0.0.1:$ApiPort"
    if ($reuseWeb) {
        Write-Host "Reusing the existing healthy AtlasLens web server at http://127.0.0.1:$WebPort"
    }
    else {
        $webArgs = @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ('"' + $webScript + '"'),
            "-WebHost", "127.0.0.1", "-WebPort", [string]$WebPort, "-ApiTarget", $apiTarget
        )
        if ($runtime.ExternalRuntime) { $webArgs += @("-RuntimeRoot", ('"' + $runtime.RuntimeRoot + '"')) }
        if ($IgnoreProjectEnv) { $webArgs += "-IgnoreProjectEnv" }
        $webProcess = Start-Process -FilePath $powershell -ArgumentList $webArgs -PassThru -WindowStyle Hidden
        Wait-AtlasLensEndpoint -Url "http://127.0.0.1:$WebPort/api/v1/capabilities" -Process $webProcess -Label "Web" -Validate {
            param($body) $null -ne $body.providers.exif -and -not [string]::IsNullOrWhiteSpace($body.version)
        }
    }

    Write-Host "AtlasLens core services are ready. Optional provider state is reported by /api/v1/capabilities."
    if ($degradedWorkerProviders.Count -gt 0) {
        Write-Warning "Optional worker degradation: $($degradedWorkerProviders -join ', ')."
    }
    Write-Host "Web: http://127.0.0.1:$WebPort"
    Write-Host "API: http://127.0.0.1:$ApiPort"
    if ($SmokeTest) {
        Write-Host "Smoke-test shutdown requested; cleaning up processes started by this command."
        return
    }
    if ($reuseApi -and $reuseWeb) {
        Write-Host "Both services were already running; leaving them active."
        return
    }
    Write-Host "Press Ctrl+C to stop processes started by this command."

    while (($reuseWeb -or -not $webProcess.HasExited) -and ($reuseApi -or -not $apiProcess.HasExited)) {
        Start-Sleep -Milliseconds 500
    }
    if (-not $reuseApi -and $apiProcess.HasExited) { throw "API process exited with code $($apiProcess.ExitCode)." }
    if (-not $reuseWeb -and $webProcess.HasExited) { throw "Web process exited with code $($webProcess.ExitCode)." }
}
finally {
    foreach ($process in @($webProcess, $apiProcess)) {
        Stop-AtlasLensProcessTree -RootProcess $process
    }
    $providersToStop = @($startedWorkerProviders)
    [array]::Reverse($providersToStop)
    foreach ($provider in $providersToStop) {
        & $powershell -NoProfile -ExecutionPolicy Bypass -File $workerStopScript -Providers $provider
        if ($LASTEXITCODE -ne 0) { Write-Warning "Cleanup for $provider exited with code $LASTEXITCODE." }
    }
}
