[CmdletBinding()]
param(
    [switch]$Execute,
    [switch]$LiveReadiness,
    [ValidateRange(0.01, 10.0)]
    [decimal]$MaxSpendUsd = 10,
    [ValidateRange(0.01, 10.0)]
    [decimal]$SoftStopUsd = 7.5,
    [ValidateRange(0.01, 10.0)]
    [decimal]$HardStopUsd = 9,
    [ValidateRange(0.01, 0.50)]
    [decimal]$MaxGpuHourlyUsd = 0.50,
    [ValidateRange(30, 345)]
    [int]$MaxWallMinutes = 345,
    [ValidateNotNullOrEmpty()]
    [string]$RuntimeRoot = "C:\AtlasLensRuntime\phase3f",
    [string]$PythonPath = "",
    [string]$SealedAcquisition = "",
    [ValidatePattern("^$|^[0-9a-f]{32}$")]
    [string]$DatasetRunId = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($env:RUNPOD_API_KEY)) {
    throw "RUNPOD_API_KEY_NOT_VISIBLE_IN_PROCESS"
}
if ($Execute -and $LiveReadiness) {
    throw "PHASE3F_OPERATOR_ACTION_INVALID"
}
if (-not ($SoftStopUsd -lt $HardStopUsd -and $HardStopUsd -lt $MaxSpendUsd)) {
    throw "PHASE3F_BUDGET_ORDER_INVALID"
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$supervisor = Join-Path $repoRoot "scripts\phase3f\supervisor.py"
$resolvedRuntimeRoot = [IO.Path]::GetFullPath($RuntimeRoot)
$operatorRoot = Join-Path $resolvedRuntimeRoot "_operator"
$operatorReceipt = Join-Path $operatorRoot "phase3f-current.json"

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

$initialRunId = $null
if (Test-Path -LiteralPath $operatorReceipt -PathType Leaf) {
    try {
        $initialRunId = (Get-Content -LiteralPath $operatorReceipt -Raw | ConvertFrom-Json).run_id
    }
    catch {
        throw "OPERATOR_RECEIPT_INVALID"
    }
}

$arguments = @(
    $supervisor,
    "--runtime-root", $resolvedRuntimeRoot,
    "--operator-receipt", $operatorReceipt,
    "--max-spend-usd", $MaxSpendUsd.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--soft-stop-usd", $SoftStopUsd.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--hard-stop-usd", $HardStopUsd.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--max-gpu-hourly-usd", $MaxGpuHourlyUsd.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--max-wall-minutes", $MaxWallMinutes.ToString([Globalization.CultureInfo]::InvariantCulture)
)
if ($Execute) {
    $arguments += "--execute"
}
if (-not [string]::IsNullOrWhiteSpace($SealedAcquisition)) {
    if ([string]::IsNullOrWhiteSpace($DatasetRunId)) {
        throw "PHASE3F_DATASET_RUN_ID_MISSING"
    }
    $resolvedSealedAcquisition = (Resolve-Path -LiteralPath $SealedAcquisition).Path
    $arguments += @(
        "--sealed-acquisition", $resolvedSealedAcquisition,
        "--dataset-run-id", $DatasetRunId
    )
}
elseif (-not [string]::IsNullOrWhiteSpace($DatasetRunId)) {
    throw "PHASE3F_SEALED_ACQUISITION_MISSING"
}
elseif ($LiveReadiness) {
    $arguments += "--live-readiness"
}

$supervisorExitCode = 1
$cleanupExitCode = 0
$supervisorOutput = @()
try {
    & $resolvedPython @arguments | Tee-Object -Variable supervisorOutput
    $supervisorExitCode = $LASTEXITCODE
}
finally {
    if ($Execute -and (Test-Path -LiteralPath $operatorReceipt -PathType Leaf)) {
        $currentReceipt = $null
        try {
            $currentReceipt = Get-Content -LiteralPath $operatorReceipt -Raw | ConvertFrom-Json
        }
        catch {
            $cleanupExitCode = 1
        }
        $needsCleanup = (
            $null -ne $currentReceipt -and (
                $currentReceipt.stage -ne "terminated" -or
                $currentReceipt.cleanup_verified -ne $true
            )
        )
        if ($needsCleanup) {
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
                (Join-Path $PSScriptRoot "stop-phase3f-runpod.ps1") `
                -RuntimeRoot $resolvedRuntimeRoot `
                -PythonPath $resolvedPython
            $cleanupExitCode = $LASTEXITCODE
        }
    }
}

if ($cleanupExitCode -ne 0) {
    throw "PHASE3F_FINALLY_CLEANUP_FAILED"
}
if ($supervisorExitCode -ne 0) {
    $typedFailure = @(
        $supervisorOutput |
            ForEach-Object { [string]$_ } |
            Where-Object { $_ -cmatch '^[A-Z][A-Z0-9_]{2,127}$' }
    ) | Select-Object -Last 1
    if (-not [string]::IsNullOrWhiteSpace([string]$typedFailure)) {
        throw [string]$typedFailure
    }
    throw "PHASE3F_SUPERVISOR_FAILED"
}
