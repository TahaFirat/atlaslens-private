[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$webDir = Join-Path $repoRoot "apps\web"
$contract = Join-Path $repoRoot "packages\contracts\openapi.yaml"
$temporaryOutput = Join-Path ([IO.Path]::GetTempPath()) ("atlaslens-openapi-{0}.d.ts" -f $PID)

if ($null -eq (Get-Command "npm.cmd" -ErrorAction SilentlyContinue)) {
    throw "npm.cmd is required on PATH."
}
if (-not (Test-Path -LiteralPath (Join-Path $webDir "node_modules"))) {
    throw "Frontend dependencies are missing. Run 'npm.cmd --prefix apps/web ci' first."
}

try {
    Push-Location $webDir
    & npm.cmd exec -- openapi-typescript $contract --output $temporaryOutput
    if ($LASTEXITCODE -ne 0) { throw "OpenAPI validation failed." }
    Write-Host "OpenAPI contract parsed successfully."
}
finally {
    Pop-Location
    Remove-Item -LiteralPath $temporaryOutput -Force -ErrorAction SilentlyContinue
}
