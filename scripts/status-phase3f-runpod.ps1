[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()]
    [string]$RuntimeRoot = "C:\AtlasLensRuntime\phase3f",
    [string]$PythonPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($env:RUNPOD_API_KEY)) {
    throw "RUNPOD_API_KEY_NOT_VISIBLE_IN_PROCESS"
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$supervisor = Join-Path $repoRoot "scripts\phase3f\supervisor.py"
$resolvedRuntimeRoot = [IO.Path]::GetFullPath($RuntimeRoot)
$operatorReceipt = Join-Path (Join-Path $resolvedRuntimeRoot "_operator") "phase3f-current.json"

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

& $resolvedPython $supervisor `
    --status `
    --runtime-root $resolvedRuntimeRoot `
    --operator-receipt $operatorReceipt
if ($LASTEXITCODE -ne 0) {
    throw "PHASE3F_STATUS_FAILED"
}
