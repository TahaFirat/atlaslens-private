[CmdletBinding()]
param(
    [switch]$RealModel,
    [string]$RealImage
)

$ErrorActionPreference = "Stop"

$webRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repoRoot = (Resolve-Path (Join-Path $webRoot "..\..")).Path
$apiRoot = Join-Path $repoRoot "services\api"
$apiPython = Join-Path $apiRoot ".venv\Scripts\python.exe"
if (-not [string]::IsNullOrWhiteSpace($env:ATLASLENS_API_PYTHON)) {
    if (-not [System.IO.Path]::IsPathRooted($env:ATLASLENS_API_PYTHON) -or
        -not (Test-Path -LiteralPath $env:ATLASLENS_API_PYTHON -PathType Leaf)) {
        throw "ATLASLENS_API_PYTHON must be a full path to an existing python.exe."
    }
    $apiPython = (Resolve-Path -LiteralPath $env:ATLASLENS_API_PYTHON).Path
}
$node = (Get-Command "node.exe" -ErrorAction Stop).Source
$vite = Join-Path $webRoot "node_modules\vite\bin\vite.js"
$apiSource = Join-Path $apiRoot "src"
$apiWorkingDirectory = Join-Path $webRoot "test-results\runtime\work\api"
New-Item -ItemType Directory -Force -Path $apiWorkingDirectory, (Join-Path $apiWorkingDirectory "data") | Out-Null
foreach ($candidate in @(
    (Join-Path $apiWorkingDirectory ".env"),
    [IO.Path]::GetFullPath((Join-Path $apiWorkingDirectory "..\..\.env"))
)) {
    if (Test-Path -LiteralPath $candidate) {
        throw "E2E validation working directory contains an env file; refusing to read it."
    }
}
$apiProcess = $null
$webProcess = $null

if (-not (Test-Path -LiteralPath $apiPython -PathType Leaf)) {
    throw "AtlasLens API virtual environment is missing."
}
if ($RealModel -and (-not $RealImage -or -not (Test-Path -LiteralPath $RealImage -PathType Leaf))) {
    throw "A licensed local image is required for the real-model E2E gate."
}

function Wait-HttpReady {
    param([string]$Url, [int]$Attempts = 60)
    for ($attempt = 0; $attempt -lt $Attempts; $attempt++) {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 1
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                return
            }
        }
        catch {
            Start-Sleep -Milliseconds 250
        }
    }
    throw "Local E2E service did not become ready: $Url"
}

try {
    $env:DATABASE_URL = "sqlite:///:memory:"
    $env:TEMP_STORAGE_DIR = (Join-Path $webRoot "test-results\api-tmp")
    $env:KEEP_UPLOADS = "false"
    $env:OCR_ENABLED = "false"
    $env:GLOBAL_MODEL_ENABLED = if ($RealModel) { "true" } else { "false" }
    $env:ALLOWED_ORIGINS = "http://127.0.0.1:5174,http://localhost:5174"
    $env:OPENAI_API_KEY = ""
    $env:OPENAI_GEO_ENABLED = "false"
    $env:PHASE5B_ENABLED = "false"
    $env:PHASE6A_ENABLED = "false"
    $env:PHASE6B_ENABLED = "false"
    $env:PHASE6C_ENABLED = "false"
    $env:G3_ENABLED = "false"
    $env:MEGALOC_ENABLED = "false"
    $env:REFERENCE_INDEX_ENABLED = "false"
    $env:MAP_EVIDENCE_ENABLED = "false"
    $env:CUSTOM_MODEL_ENABLED = "false"
    $env:RETRIEVAL_ENABLED = "false"
    $env:ATLASLENS_IGNORE_ENV_FILE = "true"
    $env:PYTHONPATH = $apiSource
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    $env:HF_DATASETS_OFFLINE = "1"
    $env:HF_HUB_DISABLE_TELEMETRY = "1"
    $env:LOG_LEVEL = "WARNING"
    $apiProcess = Start-Process `
        -FilePath $apiPython `
        -ArgumentList "-m", "uvicorn", "atlaslens_api.main:app", "--app-dir", $apiSource, "--host", "127.0.0.1", "--port", "8100" `
        -WorkingDirectory $apiWorkingDirectory `
        -WindowStyle Hidden `
        -PassThru
    Wait-HttpReady -Url "http://127.0.0.1:8100/api/v1/health"

    $env:VITE_DEV_API_TARGET = "http://127.0.0.1:8100"
    $env:VITE_API_BASE_URL = ""
    $env:VITE_MAP_PROVIDER = "maplibre"
    $env:VITE_MAP_STYLE_URL = "atlaslens://offline"
    $env:VITE_ENABLE_OPERATOR_UI = "true"
    $webProcess = Start-Process `
        -FilePath $node `
        -ArgumentList $vite, "--configLoader", "runner", "--host", "127.0.0.1", "--port", "5174" `
        -WorkingDirectory $webRoot `
        -WindowStyle Hidden `
        -PassThru
    Wait-HttpReady -Url "http://127.0.0.1:5174/"

    $env:ATLASLENS_E2E_EXTERNAL = "1"
    $env:ATLASLENS_E2E_API_URL = "http://127.0.0.1:8100"
    $env:ATLASLENS_E2E_BASE_URL = "http://127.0.0.1:5174"
    $env:ATLASLENS_REAL_MODEL_E2E = if ($RealModel) { "1" } else { "0" }
    $env:ATLASLENS_REAL_IMAGE = if ($RealModel) {
        (Resolve-Path -LiteralPath $RealImage).Path
    }
    else {
        ""
    }
    if ($RealModel) {
        & npm.cmd run test:e2e -- --reporter=list -g "real no-EXIF"
    }
    else {
        & npm.cmd run test:e2e -- --reporter=list
    }
    exit $LASTEXITCODE
}
finally {
    foreach ($process in @($webProcess, $apiProcess)) {
        if ($null -ne $process -and -not $process.HasExited) {
            Stop-Process -Id $process.Id -Force
            $null = $process.WaitForExit(5000)
        }
    }
}
