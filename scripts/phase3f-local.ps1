[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("AcquireOnly", "DiagnoseAcquisition", "Status", "Resume", "ComputeOnly", "Cleanup")]
    [string]$Action,
    [ValidateNotNullOrEmpty()]
    [string]$RuntimeRoot = "D:\AtlasLensRuntime\phase3f-local",
    [ValidateRange(30, 720)]
    [int]$MaxWallMinutes = 720,
    [string]$PythonPath = "",
    [switch]$Execute,
    [string]$ConfirmRunId = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$entrypoint = Join-Path $repoRoot "scripts\phase3f\local_first.py"
$resolvedRuntimeRoot = [IO.Path]::GetFullPath($RuntimeRoot)
$runtimeDrive = [IO.Path]::GetPathRoot($resolvedRuntimeRoot)
if ($runtimeDrive -eq "C:\") {
    throw "LOCAL_RUNTIME_C_DRIVE_REFUSED"
}

if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $PythonPath = "C:\AtlasLensRuntime\api-venv\Scripts\python.exe"
}
$resolvedPython = (Resolve-Path -LiteralPath $PythonPath).Path

if ($Action -in @("AcquireOnly", "Resume", "DiagnoseAcquisition")) {
    if ([string]::IsNullOrWhiteSpace($env:MAPILLARY_ACCESS_TOKEN)) {
        throw "MAPILLARY_ACCESS_TOKEN_MISSING"
    }
}
if ($Action -in @("AcquireOnly", "Resume")) {
    $driveInfo = [IO.DriveInfo]::new($runtimeDrive)
    if ($driveInfo.AvailableFreeSpace -lt 8GB) {
        throw "LOCAL_RUNTIME_FREE_SPACE_BELOW_8_GIB"
    }
    [IO.Directory]::CreateDirectory($resolvedRuntimeRoot) | Out-Null
}
elseif (-not (Test-Path -LiteralPath $resolvedRuntimeRoot -PathType Container)) {
    throw "LOCAL_RUNTIME_ROOT_MISSING"
}

$tempRoot = Join-Path $resolvedRuntimeRoot "_tmp"
[IO.Directory]::CreateDirectory($tempRoot) | Out-Null
$env:TEMP = $tempRoot
$env:TMP = $tempRoot
$env:PYTHONDONTWRITEBYTECODE = "1"

$arguments = @(
    $entrypoint,
    "--repository-root", $repoRoot,
    "--runtime-root", $resolvedRuntimeRoot,
    "--max-wall-minutes", $MaxWallMinutes.ToString()
)

switch ($Action) {
    "AcquireOnly" {
        $arguments = @($entrypoint, "acquire-only") + $arguments[1..($arguments.Length - 1)]
    }
    "Resume" {
        $arguments = @($entrypoint, "resume") + $arguments[1..($arguments.Length - 1)]
    }
    "DiagnoseAcquisition" {
        $arguments = @($entrypoint, "diagnose-acquisition") + $arguments[1..($arguments.Length - 1)]
    }
    "Status" {
        $arguments = @($entrypoint, "status") + $arguments[1..($arguments.Length - 1)]
    }
    "ComputeOnly" {
        Remove-Item Env:MAPILLARY_ACCESS_TOKEN -ErrorAction SilentlyContinue
        $env:HF_HUB_OFFLINE = "1"
        $env:TRANSFORMERS_OFFLINE = "1"
        $arguments = @($entrypoint, "compute-only") + $arguments[1..($arguments.Length - 1)]
        $arguments += @(
            "--model", (Join-Path $repoRoot ".local\models\phase6c\megaloc\model.safetensors"),
            "--vendor-root", (Join-Path $repoRoot ".local\vendor\megaloc")
        )
    }
    "Cleanup" {
        $arguments = @($entrypoint, "cleanup") + $arguments[1..($arguments.Length - 1)]
        if ($Execute) {
            if ([string]::IsNullOrWhiteSpace($ConfirmRunId)) {
                throw "LOCAL_CLEANUP_CONFIRMATION_MISSING"
            }
            $arguments += @("--execute", "--confirm-run-id", $ConfirmRunId)
        }
    }
}

& $resolvedPython @arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
