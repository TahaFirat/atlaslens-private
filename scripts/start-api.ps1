[CmdletBinding()]
param(
    [string]$ApiHost = "127.0.0.1",
    [int]$ApiPort = $(if ($env:API_PORT) { [int]$env:API_PORT } else { 8000 }),
    [AllowEmptyString()][string]$RuntimeRoot = $(if ($env:ATLASLENS_RUNTIME_ROOT) { $env:ATLASLENS_RUNTIME_ROOT } else { "" }),
    [switch]$IgnoreProjectEnv,
    [switch]$SkipMigration
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$apiDir = Join-Path $repoRoot "services\api"
$alembicConfig = Join-Path $apiDir "alembic.ini"
$runtimeSupport = Join-Path $PSScriptRoot "runtime\AtlasLensRuntime.ps1"
. $runtimeSupport
$runtime = Get-AtlasLensRuntimeConfiguration -RepositoryRoot $repoRoot -RuntimeRoot $RuntimeRoot
$environmentSnapshot = Set-AtlasLensRuntimeEnvironment -Configuration $runtime -IncludeCaches -IncludeOfflineModelGuards
$workingDirectory = $apiDir
if ($IgnoreProjectEnv) {
    if (-not $runtime.ExternalRuntime) {
        throw "IgnoreProjectEnv requires an explicit external RuntimeRoot."
    }
    $validationRoot = Join-Path $runtime.RuntimeRoot "validation"
    $workingDirectory = Join-Path $validationRoot "work\api"
    New-Item -ItemType Directory -Force -Path $workingDirectory, (Join-Path $workingDirectory "data") | Out-Null
    foreach ($candidate in @(
        (Join-Path $workingDirectory ".env"),
        [IO.Path]::GetFullPath((Join-Path $workingDirectory "..\..\.env"))
    )) {
        if (Test-Path -LiteralPath $candidate) {
            throw "Validation working directory contains an env file; refusing to read it."
        }
    }
}

try {
    Assert-AtlasLensApiRuntime $runtime
    Push-Location $workingDirectory
    try {
        if (-not $SkipMigration) {
            & $runtime.ApiPython -m alembic -c $alembicConfig upgrade head
            if ($LASTEXITCODE -ne 0) { throw "Database migration failed with code $LASTEXITCODE." }
        }

        Write-Host "Starting AtlasLens API at http://${ApiHost}:$ApiPort"
        & $runtime.ApiPython -m uvicorn atlaslens_api.main:app --app-dir $runtime.ApiSourceDirectory --host $ApiHost --port $ApiPort
        if ($LASTEXITCODE -ne 0) { throw "AtlasLens API exited with code $LASTEXITCODE." }
    }
    finally { Pop-Location }
}
finally { Restore-AtlasLensRuntimeEnvironment $environmentSnapshot }
