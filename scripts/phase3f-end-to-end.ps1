[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Preflight", "CloudPlan", "RemoteEnvironmentPlan", "TrainingPlan", "ReconcileLocalReceipts", "Execute", "Status", "Resume", "EmergencyStop", "Cleanup")]
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

function Get-SanitizedChildFailure {
    param(
        [Parameter(Mandatory = $true)][object[]]$ChildOutput,
        [Parameter(Mandatory = $true)][string]$FallbackCode
    )
    for ($index = $ChildOutput.Count - 1; $index -ge 0; $index--) {
        $line = [string]$ChildOutput[$index]
        if ($line -cmatch '^[A-Z][A-Z0-9_]{2,127}$') {
            return [PSCustomObject]@{ Code = $line; Diagnostic = $null }
        }
        try {
            $document = $line | ConvertFrom-Json -ErrorAction Stop
            $code = [string]$document.error_code
            if ($code -cnotmatch '^[A-Z][A-Z0-9_]{2,127}$') {
                continue
            }
            $safe = [ordered]@{ error_code = $code }
            foreach ($name in @("artifact", "field")) {
                $value = [string]$document.$name
                if (-not [string]::IsNullOrWhiteSpace($value) -and $value -cmatch '^[A-Za-z0-9_.-]{1,128}$') {
                    $safe[$name] = $value
                }
            }
            foreach ($name in @("expected", "actual")) {
                $value = $document.$name
                if (
                    $value -is [bool] -or
                    $value -is [int] -or
                    ($value -is [string] -and [string]$value -cmatch '^[a-f0-9]{64}$')
                ) {
                    $safe[$name] = $value
                }
            }
            return [PSCustomObject]@{
                Code = $code
                Diagnostic = ($safe | ConvertTo-Json -Compress)
            }
        }
        catch {
            continue
        }
    }
    return [PSCustomObject]@{ Code = $FallbackCode; Diagnostic = $null }
}

function Throw-SanitizedChildFailure {
    param(
        [Parameter(Mandatory = $true)][object[]]$ChildOutput,
        [Parameter(Mandatory = $true)][string]$FallbackCode
    )
    $failure = Get-SanitizedChildFailure `
        -ChildOutput $ChildOutput `
        -FallbackCode $FallbackCode
    if (-not [string]::IsNullOrWhiteSpace([string]$failure.Diagnostic)) {
        [Console]::Error.WriteLine([string]$failure.Diagnostic)
    }
    throw [string]$failure.Code
}

function Convert-ControlResult {
    param(
        [Parameter(Mandatory = $true)][object[]]$ControlOutput,
        [Parameter(Mandatory = $true)][string]$ExpectedAction
    )
    if ($ControlOutput.Count -ne 1) {
        throw "PHASE3F_CONTROL_OUTPUT_INVALID"
    }
    try {
        $document = $ControlOutput[0] | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "PHASE3F_CONTROL_OUTPUT_INVALID"
    }
    if ([string]$document.action -ne $ExpectedAction) {
        throw "PHASE3F_CONTROL_OUTPUT_INVALID"
    }
    return $document
}

function Invoke-Control {
    param([Parameter(Mandatory = $true)][string]$ControlAction)
    $arguments = @(
        $control,
        $ControlAction,
        "--repository-root", $repoRoot,
        "--runtime-root", $resolvedRuntime
    )
    if ($ControlAction -in @("resume-plan", "cloud-plan", "training-plan", "reconcile-local-receipts", "status")) {
        $arguments += @("--cloud-runtime-root", $resolvedCloudRuntime)
    }
    $controlOutput = @(& $resolvedPython @arguments)
    if ($LASTEXITCODE -ne 0) {
        Throw-SanitizedChildFailure `
            -ChildOutput $controlOutput `
            -FallbackCode "PHASE3F_$($ControlAction.ToUpperInvariant().Replace('-', '_'))_CHILD_FAILED"
    }
    return $controlOutput
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
                Throw-SanitizedChildFailure `
                    -ChildOutput $localOutput `
                    -FallbackCode "PHASE3F_LOCAL_ACQUISITION_CHILD_FAILED"
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
            $cloudOutput = @(& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $cloudLauncher `
                -Execute `
                -RuntimeRoot $resolvedCloudRuntime `
                -PythonPath $resolvedPython `
                -SealedAcquisition $SealedRoot `
                -DatasetRunId $RunId `
                -MaxSpendUsd $MaxSpendUsd `
                -SoftStopUsd $SoftStopUsd `
                -HardStopUsd $HardStopUsd `
                -MaxGpuHourlyUsd $MaxGpuHourlyUsd `
                -MaxWallMinutes $TrainingMaxWallMinutes)
            if ($LASTEXITCODE -ne 0) {
                Throw-SanitizedChildFailure `
                    -ChildOutput $cloudOutput `
                    -FallbackCode "PHASE3F_CLOUD_TRAINING_CHILD_FAILED"
            }
            $cloudOutput
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
    "CloudPlan" {
        Invoke-Control -ControlAction "cloud-plan"
    }
    "RemoteEnvironmentPlan" {
        Invoke-Control -ControlAction "remote-environment-plan"
    }
    "TrainingPlan" {
        Invoke-Control -ControlAction "training-plan"
    }
    "ReconcileLocalReceipts" {
        Invoke-Control -ControlAction "reconcile-local-receipts"
    }
    { $_ -in @("Execute", "Resume") } {
        $null = Invoke-Control -ControlAction "preflight"
        $currentPath = Join-Path $resolvedRuntime "current.json"
        $resumeExisting = Test-Path -LiteralPath $currentPath -PathType Leaf
        if ($Action -eq "Resume" -and -not $resumeExisting) {
            throw "PHASE3F_RESUME_STATE_MISSING"
        }
        $plan = $null
        if ($resumeExisting) {
            $planOutput = @(Invoke-Control -ControlAction "resume-plan")
            $plan = Convert-ControlResult `
                -ControlOutput $planOutput `
                -ExpectedAction "resume-plan"
        }

        $acquisitionRequired = -not $resumeExisting -or $plan.acquisition_required -eq $true
        if ($acquisitionRequired) {
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
            $null = Invoke-Control -ControlAction "readiness"
            $planOutput = @(Invoke-Control -ControlAction "resume-plan")
            $plan = Convert-ControlResult `
                -ControlOutput $planOutput `
                -ExpectedAction "resume-plan"
        }

        if ([string]$plan.next_phase -in @("training_running", "training_completed")) {
            $plan | ConvertTo-Json -Compress -Depth 4
            break
        }
        if (
            $plan.acquisition_required -ne $false -or
            $plan.mapillary_required -ne $false -or
            [string]$plan.next_phase -ne "cloud_inventory"
        ) {
            throw "PHASE3F_RESUME_PLAN_INVALID"
        }
        $cloudPlanOutput = @(Invoke-Control -ControlAction "cloud-plan")
        $cloudPlan = Convert-ControlResult `
            -ControlOutput $cloudPlanOutput `
            -ExpectedAction "cloud-plan"
        if (
            $cloudPlan.ready_for_live_inventory -ne $true -or
            $cloudPlan.ready_for_create_after_live_gates -ne $true -or
            @($cloudPlan.local_blockers).Count -ne 0 -or
            [int]$cloudPlan.archive_conflicts -ne 0 -or
            [int]$cloudPlan.active_local_receipts -ne 0 -or
            [int]$cloudPlan.unclean_local_receipts -ne 0 -or
            [int]$cloudPlan.cloud_mutations -ne 0 -or
            [int]$cloudPlan.runpod_api_calls -ne 0
        ) {
            throw "PHASE3F_CLOUD_PLAN_BLOCKED"
        }
        $runId = [string]$plan.run_id
        $sealedRoot = Join-Path (Join-Path $resolvedRuntime $runId) "sealed-acquisition"
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
