[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Preflight", "Execute", "Status", "Resume", "EmergencyStop", "Cleanup")]
    [string]$Action,
    [ValidateNotNullOrEmpty()]
    [string]$RuntimeRoot = "D:\AtlasLensRuntime\phase3f-local",
    [ValidateNotNullOrEmpty()]
    [string]$CloudRuntimeRoot = "C:\AtlasLensRuntime\phase3f",
    [ValidateRange(30, 720)]
    [int]$AcquisitionMaxWallMinutes = 720,
    [ValidateRange(30, 345)]
    [int]$TrainingMaxWallMinutes = 345,
    [ValidateRange(0.01, 3.0)]
    [decimal]$MaxSpendUsd = 3,
    [ValidateRange(0.01, 2.95)]
    [decimal]$SoftStopUsd = 2.95,
    [ValidateRange(0.01, 2.99)]
    [decimal]$HardStopUsd = 2.99,
    [ValidateRange(0.01, 0.50)]
    [decimal]$MaxGpuHourlyUsd = 0.50,
    [string]$PythonPath = "",
    [switch]$CloudConsent,
    [switch]$ExecuteCleanup,
    [string]$ConfirmRunId = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$control = Join-Path $repoRoot "scripts\phase3f\end_to_end.py"
$localLauncher = Join-Path $repoRoot "scripts\phase3f-local.ps1"
$cloudLauncher = Join-Path $repoRoot "scripts\start-phase3f-runpod.ps1"
$cloudStopper = Join-Path $repoRoot "scripts\stop-phase3f-runpod.ps1"
$resolvedRuntime = [IO.Path]::GetFullPath($RuntimeRoot)
$resolvedCloudRuntime = [IO.Path]::GetFullPath($CloudRuntimeRoot)

if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $runtimePython = "C:\AtlasLensRuntime\api-venv\Scripts\python.exe"
    $repositoryPython = Join-Path $repoRoot "services\api\.venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $runtimePython -PathType Leaf) {
        $PythonPath = $runtimePython
    }
    else {
        $PythonPath = $repositoryPython
    }
}
$resolvedPython = (Resolve-Path -LiteralPath $PythonPath).Path

function Invoke-Control {
    param([Parameter(Mandatory = $true)][string]$ControlAction)
    $arguments = @(
        $control,
        $ControlAction,
        "--repository-root", $repoRoot,
        "--runtime-root", $resolvedRuntime
    )
    if ($ControlAction -eq "status") {
        $arguments += @("--cloud-runtime-root", $resolvedCloudRuntime)
    }
    & $resolvedPython @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "PHASE3F_$($ControlAction.ToUpperInvariant())_FAILED"
    }
}

function Invoke-WithSecureEnvironment {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][Security.SecureString]$Secret,
        [Parameter(Mandatory = $true)][scriptblock]$Operation
    )
    $pointer = [IntPtr]::Zero
    $plain = $null
    try {
        $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secret)
        $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
        if ([string]::IsNullOrWhiteSpace($plain)) {
            throw "PHASE3F_SECRET_EMPTY"
        }
        [Environment]::SetEnvironmentVariable($Name, $plain, "Process")
        & $Operation
    }
    finally {
        [Environment]::SetEnvironmentVariable($Name, $null, "Process")
        $plain = $null
        if ($pointer -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
        }
    }
}

function Invoke-LocalAcquisition {
    param([Parameter(Mandatory = $true)][bool]$ResumeExisting)
    $mapillarySecret = Read-Host "Mapillary API token (local acquisition only)" -AsSecureString
    try {
        Invoke-WithSecureEnvironment -Name "MAPILLARY_ACCESS_TOKEN" -Secret $mapillarySecret -Operation {
            $localAction = if ($ResumeExisting) { "Resume" } else { "AcquireOnly" }
            $localOutput = @(& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $localLauncher `
                -Action $localAction `
                -RuntimeRoot $resolvedRuntime `
                -MaxWallMinutes $AcquisitionMaxWallMinutes `
                -PythonPath $resolvedPython)
            if ($LASTEXITCODE -ne 0) {
                throw "PHASE3F_LOCAL_ACQUISITION_FAILED"
            }
            if ($localOutput.Count -ne 1) {
                throw "PHASE3F_LOCAL_ACQUISITION_OUTPUT_INVALID"
            }
            try {
                return ($localOutput[0] | ConvertFrom-Json -ErrorAction Stop)
            }
            catch {
                throw "PHASE3F_LOCAL_ACQUISITION_OUTPUT_INVALID"
            }
        }
    }
    finally {
        $mapillarySecret.Dispose()
        Remove-Item Env:MAPILLARY_ACCESS_TOKEN -ErrorAction SilentlyContinue
    }
}

function Invoke-CloudTraining {
    param(
        [Parameter(Mandatory = $true)][string]$RunId,
        [Parameter(Mandatory = $true)][string]$SealedRoot
    )
    if (-not $CloudConsent) {
        throw "PHASE3F_EXPLICIT_CLOUD_CONSENT_REQUIRED"
    }
    Remove-Item Env:MAPILLARY_ACCESS_TOKEN -ErrorAction SilentlyContinue
    $runPodSecret = Read-Host "RunPod API key (cloud lifecycle only)" -AsSecureString
    try {
        Invoke-WithSecureEnvironment -Name "RUNPOD_API_KEY" -Secret $runPodSecret -Operation {
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $cloudLauncher `
                -Execute `
                -RuntimeRoot $resolvedCloudRuntime `
                -PythonPath $resolvedPython `
                -SealedAcquisition $SealedRoot `
                -DatasetRunId $RunId `
                -MaxSpendUsd $MaxSpendUsd `
                -SoftStopUsd $SoftStopUsd `
                -HardStopUsd $HardStopUsd `
                -MaxGpuHourlyUsd $MaxGpuHourlyUsd `
                -MaxWallMinutes $TrainingMaxWallMinutes
            if ($LASTEXITCODE -ne 0) {
                throw "PHASE3F_CLOUD_TRAINING_FAILED"
            }
        }
    }
    finally {
        $runPodSecret.Dispose()
        Remove-Item Env:RUNPOD_API_KEY -ErrorAction SilentlyContinue
        Remove-Item Env:MAPILLARY_ACCESS_TOKEN -ErrorAction SilentlyContinue
    }
}

switch ($Action) {
    "Preflight" {
        Invoke-Control -ControlAction "preflight"
    }
    "Status" {
        Invoke-Control -ControlAction "status"
    }
    { $_ -in @("Execute", "Resume") } {
        $null = Invoke-Control -ControlAction "preflight"
        $currentPath = Join-Path $resolvedRuntime "current.json"
        $resumeExisting = Test-Path -LiteralPath $currentPath -PathType Leaf
        if ($Action -eq "Resume" -and -not $resumeExisting) {
            throw "PHASE3F_RESUME_STATE_MISSING"
        }
        $localResult = Invoke-LocalAcquisition -ResumeExisting $resumeExisting
        if (
            [string]$localResult.stage -like "PAUSED_*" -or
            [string]$localResult.stage -in @(
                "MEDIA_SPLIT_MINIMUM_UNAVAILABLE",
                "MEDIA_CORPUS_EXHAUSTED",
                "MEDIA_REPLENISHMENT_LIMIT_REACHED"
            )
        ) {
            $localResult | ConvertTo-Json -Compress -Depth 4
            break
        }
        Invoke-Control -ControlAction "readiness"
        $current = Get-Content -LiteralPath $currentPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $runId = [string]$current.run_id
        if ($runId -notmatch '^[0-9a-f]{32}$') {
            throw "PHASE3F_CURRENT_RUN_INVALID"
        }
        $sealedRoot = Join-Path (Join-Path $resolvedRuntime $runId) "sealed-acquisition"
        if (-not (Test-Path -LiteralPath $sealedRoot -PathType Container)) {
            throw "DATASET_NOT_READY_FOR_TRAINING"
        }
        Invoke-CloudTraining -RunId $runId -SealedRoot $sealedRoot
        Invoke-Control -ControlAction "status"
    }
    "EmergencyStop" {
        $runPodSecret = Read-Host "RunPod API key (emergency termination only)" -AsSecureString
        try {
            Invoke-WithSecureEnvironment -Name "RUNPOD_API_KEY" -Secret $runPodSecret -Operation {
                & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $cloudStopper `
                    -RuntimeRoot $resolvedCloudRuntime `
                    -PythonPath $resolvedPython
                if ($LASTEXITCODE -ne 0) {
                    throw "PHASE3F_EMERGENCY_STOP_FAILED"
                }
            }
        }
        finally {
            $runPodSecret.Dispose()
            Remove-Item Env:RUNPOD_API_KEY -ErrorAction SilentlyContinue
        }
    }
    "Cleanup" {
        $operatorReceipt = Join-Path (Join-Path $resolvedCloudRuntime "_operator") "phase3f-current.json"
        if (Test-Path -LiteralPath $operatorReceipt -PathType Leaf) {
            $operator = Get-Content -LiteralPath $operatorReceipt -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($operator.stage -ne "terminated" -or $operator.cleanup_verified -ne $true) {
                throw "PHASE3F_CLOUD_CLEANUP_NOT_VERIFIED_USE_EMERGENCY_STOP"
            }
        }
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $localLauncher `
            -Action Cleanup `
            -RuntimeRoot $resolvedRuntime `
            -PythonPath $resolvedPython `
            -Execute:$ExecuteCleanup `
            -ConfirmRunId $ConfirmRunId
        if ($LASTEXITCODE -ne 0) {
            throw "PHASE3F_LOCAL_CLEANUP_FAILED"
        }
    }
}
