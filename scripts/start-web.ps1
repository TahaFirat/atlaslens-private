[CmdletBinding()]
param(
    [string]$WebHost = "127.0.0.1",
    [int]$WebPort = $(if ($env:VITE_DEV_PORT) { [int]$env:VITE_DEV_PORT } else { 5173 }),
    [string]$ApiTarget = $(if ($env:VITE_DEV_API_TARGET) { $env:VITE_DEV_API_TARGET } else { "http://127.0.0.1:8000" }),
    [AllowEmptyString()][string]$RuntimeRoot = $(if ($env:ATLASLENS_RUNTIME_ROOT) { $env:ATLASLENS_RUNTIME_ROOT } else { "" }),
    [switch]$IgnoreProjectEnv
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$webDir = Join-Path $repoRoot "apps\web"
$runtimeSupport = Join-Path $PSScriptRoot "runtime\AtlasLensRuntime.ps1"
. $runtimeSupport
$runtime = Get-AtlasLensRuntimeConfiguration -RepositoryRoot $repoRoot -RuntimeRoot $RuntimeRoot
$environmentSnapshot = Set-AtlasLensRuntimeEnvironment -Configuration $runtime -IncludeCaches
if ($IgnoreProjectEnv) {
    Set-AtlasLensProcessVariable -Snapshot $environmentSnapshot -Name "ATLASLENS_IGNORE_ENV_FILE" -Value "true"
}

try {
    if ($null -eq (Get-Command "npm.cmd" -ErrorAction SilentlyContinue)) {
        throw "Required command 'npm.cmd' was not found on PATH."
    }
    if (-not (Test-Path -LiteralPath (Join-Path $webDir "node_modules\vite\package.json") -PathType Leaf)) {
        throw "Frontend dependencies are missing. Restore them with: $(Get-AtlasLensWebRestoreCommand $runtime)"
    }

    $env:VITE_API_BASE_URL = ""
    $env:VITE_DEV_API_TARGET = $ApiTarget.TrimEnd("/")
    $env:VITE_DEV_PORT = [string]$WebPort
    Write-Host "Starting AtlasLens web at http://${WebHost}:$WebPort (API proxy: $env:VITE_DEV_API_TARGET)"
    & npm.cmd --prefix $webDir run dev -- --host $WebHost --port $WebPort
    if ($LASTEXITCODE -ne 0) { throw "AtlasLens web exited with code $LASTEXITCODE." }
}
finally { Restore-AtlasLensRuntimeEnvironment $environmentSnapshot }
