[CmdletBinding()]
param(
    [switch]$SkipDockerValidation,
    [switch]$SkipE2E,
    [switch]$SkipInstall,
    [AllowEmptyString()][string]$RuntimeRoot = $(if ($env:ATLASLENS_RUNTIME_ROOT) { $env:ATLASLENS_RUNTIME_ROOT } else { "" })
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$apiDir = Join-Path $repoRoot "services\api"
$webDir = Join-Path $repoRoot "apps\web"
$runtimeSupport = Join-Path $PSScriptRoot "runtime\AtlasLensRuntime.ps1"
. $runtimeSupport
$runtime = Get-AtlasLensRuntimeConfiguration -RepositoryRoot $repoRoot -RuntimeRoot $RuntimeRoot
$environmentSnapshot = Set-AtlasLensRuntimeEnvironment -Configuration $runtime -IncludeCaches -IncludeOfflineModelGuards

try {
    Set-AtlasLensProcessVariable -Snapshot $environmentSnapshot -Name "ATLASLENS_API_PYTHON" -Value $runtime.ApiPython

    if ($null -eq (Get-Command "npm.cmd" -ErrorAction SilentlyContinue)) {
        throw "Required command 'npm.cmd' was not found on PATH."
    }
    if (-not $SkipInstall -and $null -eq (Get-Command "uv" -ErrorAction SilentlyContinue)) {
        throw "Required command 'uv' was not found on PATH."
    }

    if (-not $SkipInstall) {
        Assert-AtlasLensDiskBudget
        if ($runtime.ExternalRuntime) {
            New-Item -ItemType Directory -Force -Path $runtime.RuntimeRoot, $runtime.UvCache, $runtime.NpmCache | Out-Null
        }

        $uvLock = Join-Path $apiDir "uv.lock"
        $uvLockHash = (Get-FileHash -LiteralPath $uvLock -Algorithm SHA256).Hash
        & uv sync --project $apiDir --frozen --all-groups
        if ($LASTEXITCODE -ne 0) { throw "Backend dependency installation failed." }
        if ((Get-FileHash -LiteralPath $uvLock -Algorithm SHA256).Hash -ne $uvLockHash) {
            throw "Frozen backend restoration unexpectedly changed uv.lock."
        }
        Assert-AtlasLensDiskBudget

        $packageLock = Join-Path $webDir "package-lock.json"
        $packageLockHash = (Get-FileHash -LiteralPath $packageLock -Algorithm SHA256).Hash
        $npmArguments = @("--prefix", $webDir, "ci", "--no-audit", "--no-fund")
        if ($runtime.ExternalRuntime) { $npmArguments += @("--cache", $runtime.NpmCache) }
        & npm.cmd @npmArguments
        if ($LASTEXITCODE -ne 0) { throw "Frontend dependency installation failed." }
        if ((Get-FileHash -LiteralPath $packageLock -Algorithm SHA256).Hash -ne $packageLockHash) {
            throw "Frozen frontend restoration unexpectedly changed package-lock.json."
        }
        Assert-AtlasLensDiskBudget
    }

    Assert-AtlasLensApiRuntime $runtime
    if (-not (Test-Path -LiteralPath (Join-Path $webDir "node_modules") -PathType Container)) {
        throw "Frontend dependencies are missing. Restore them with: $(Get-AtlasLensWebRestoreCommand $runtime)"
    }

    Push-Location $apiDir
    try {
        & $runtime.ApiPython -m ruff check .
        if ($LASTEXITCODE -ne 0) { throw "Backend lint failed." }
        & $runtime.ApiPython -m mypy
        if ($LASTEXITCODE -ne 0) { throw "Backend type checking failed." }
        & $runtime.ApiPython -m pytest
        if ($LASTEXITCODE -ne 0) { throw "Backend tests failed." }
    }
    finally { Pop-Location }

    Push-Location $webDir
    try {
        foreach ($script in @("lint", "typecheck", "test", "build", "check:api")) {
            & npm.cmd run $script
            if ($LASTEXITCODE -ne 0) { throw "Frontend script '$script' failed." }
        }
        if (-not $SkipE2E -and (Test-Path -LiteralPath (Join-Path $webDir "playwright.config.ts"))) {
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $webDir "scripts\run-e2e-local.ps1")
            if ($LASTEXITCODE -ne 0) { throw "End-to-end tests failed." }
        }
    }
    finally { Pop-Location }

    & (Join-Path $PSScriptRoot "validate-contract.ps1")

    if (-not $SkipDockerValidation) {
        if ($null -eq (Get-Command "docker" -ErrorAction SilentlyContinue)) {
            throw "Docker is required for Compose validation. Re-run with -SkipDockerValidation only when recording this environmental blocker."
        }
        $previousPassword = $env:POSTGRES_PASSWORD
        try {
            $env:POSTGRES_PASSWORD = "compose-validation-only"
            & docker compose -f (Join-Path $repoRoot "docker-compose.yml") config --quiet
            if ($LASTEXITCODE -ne 0) { throw "Docker Compose validation failed." }
        }
        finally { $env:POSTGRES_PASSWORD = $previousPassword }
    }

    Write-Host "All requested checks passed."
}
finally { Restore-AtlasLensRuntimeEnvironment $environmentSnapshot }
