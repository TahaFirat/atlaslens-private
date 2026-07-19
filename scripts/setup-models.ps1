[CmdletBinding()]
param(
    [string]$CacheRoot = $(if ($env:ATLAS_MODEL_CACHE) { $env:ATLAS_MODEL_CACHE } else { Join-Path $env:LOCALAPPDATA "AtlasLens\models" }),
    [Parameter(Mandatory = $true)][string]$SmokeImage
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$apiDir = Join-Path $repoRoot "services\api"
if ($null -eq (Get-Command "uv" -ErrorAction SilentlyContinue)) { throw "Required command 'uv' was not found." }
$probe = $CacheRoot
while (-not (Test-Path -LiteralPath $probe)) {
    $parent = Split-Path $probe -Parent
    if (-not $parent -or $parent -eq $probe) { break }
    $probe = $parent
}
$drive = (Get-Item -LiteralPath $probe -ErrorAction SilentlyContinue).PSDrive
if ($drive -and $drive.Free -lt 4GB) { throw "At least 4 GB free disk space is required." }
if (-not (Test-Path -LiteralPath $SmokeImage -PathType Leaf)) { throw "A licensed smoke-test image is required." }
Write-Host "Installing locked AtlasLens model dependencies..."
& uv sync --project $apiDir --frozen
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
Write-Host "Installing verified official GeoCLIP artifacts..."
& uv run --project $apiDir python -m atlaslens_api.model_management.cli --cache-root $CacheRoot install geoclip
if ($LASTEXITCODE -ne 0) { throw "GeoCLIP artifact installation failed." }
& uv run --project $apiDir python -m atlaslens_api.model_management.cli --cache-root $CacheRoot verify geoclip
if ($LASTEXITCODE -ne 0) { throw "GeoCLIP verification failed." }
& uv run --project $apiDir python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback')"
& uv run --project $apiDir python -m atlaslens_api.model_management.cli --cache-root $CacheRoot test geoclip --image $SmokeImage --device auto
if ($LASTEXITCODE -ne 0) { throw "GeoCLIP smoke inference failed." }
Write-Host "Next: uv run --project services/api uvicorn atlaslens_api.main:app --host 127.0.0.1 --port 8000"
