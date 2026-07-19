[CmdletBinding()]
param(
    [string]$Image,
    [switch]$UseOpenAI,
    [switch]$CloudConsent
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot "services\api\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Backend environment is missing. Run the documented API setup command first."
}
if ($UseOpenAI -and -not $CloudConsent) {
    throw "-UseOpenAI requires the separate explicit -CloudConsent flag."
}
if ([string]::IsNullOrWhiteSpace($Image)) {
    $reviewedRoot = Join-Path $ProjectRoot ".local\acceptance-images"
    $candidate = Get-ChildItem -LiteralPath $reviewedRoot -File -Filter "*.jpg" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $candidate) {
        throw "Pass -Image with a licensed user-provided image; no reviewed local smoke asset is available."
    }
    $Image = $candidate.FullName
}

$rapidRoot = if ($env:RAPIDOCR_MODEL_ROOT) {
    $env:RAPIDOCR_MODEL_ROOT
}
elseif ($env:LOCALAPPDATA) {
    Join-Path $env:LOCALAPPDATA "AtlasLens\models\rapidocr-3.9.1"
}
else {
    Join-Path $HOME ".atlaslens\models\rapidocr-3.9.1"
}
& $Python (Join-Path $PSScriptRoot "verify_rapidocr_runtime.py") --project-root $ProjectRoot --model-root $rapidRoot
if ($LASTEXITCODE -ne 0) { throw "RapidOCR real-inference verification failed." }

& $Python (Join-Path $PSScriptRoot "verify_phase6b_models.py") --project-root $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw "Phase 6B diagnostics failed." }

$Arguments = @(
    (Join-Path $PSScriptRoot "smoke_phase6b.py"),
    "--project-root", $ProjectRoot,
    "--image", (Resolve-Path -LiteralPath $Image).Path
)
if ($UseOpenAI) {
    $Arguments += "--use-openai"
    $Arguments += "--cloud-consent"
}
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Phase 6B smoke test failed." }
