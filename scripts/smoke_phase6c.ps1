[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Image,
    [string]$Output = ".local\evaluation\smoke\phase6c-target-final.json",
    [string]$ScoreOutput = ".local\evaluation\smoke\phase6c-target-final-score.json",
    [string]$ReferenceIndex = ".local\indexes\turkiye-megaloc",
    [ValidateRange(1024, 65535)]
    [int]$Port = 8760
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot "services\api\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Backend environment is missing. Run the documented API setup first."
}
$ResolvedImage = (Resolve-Path -LiteralPath $Image).Path
$ResolvedIndex = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot $ReferenceIndex)).Path
$ResolvedOutput = [IO.Path]::GetFullPath((Join-Path $ProjectRoot $Output))
$ResolvedScoreOutput = [IO.Path]::GetFullPath((Join-Path $ProjectRoot $ScoreOutput))
if (Test-Path -LiteralPath $ResolvedOutput) {
    throw "The requested immutable smoke report already exists; choose a new output path."
}
if (Test-Path -LiteralPath $ResolvedScoreOutput) {
    throw "The requested immutable score report already exists; choose a new output path."
}

& $Python (Join-Path $PSScriptRoot "verify_megaloc_reference_index.py") --index $ResolvedIndex
if ($LASTEXITCODE -ne 0) { throw "The leakage-attested reference index is not ready." }
& $Python (Join-Path $PSScriptRoot "verify_phase6c_workers.py") --port 8794 --device cuda
if ($LASTEXITCODE -ne 0) { throw "The MegaLoc worker is not ready for real inference." }

$RunRoot = Join-Path $ProjectRoot ".local\phase6c-smoke"
$DatabasePath = Join-Path $RunRoot "atlaslens.sqlite3"
$TemporaryRoot = Join-Path $RunRoot "tmp"
New-Item -ItemType Directory -Force -Path $RunRoot, $TemporaryRoot | Out-Null
if (Test-Path -LiteralPath $DatabasePath) {
    throw "The Phase 6C smoke database already exists; archive it or choose a clean checkout."
}

$EnvironmentOverrides = [ordered]@{
    "DATABASE_URL" = "sqlite:///$($DatabasePath.Replace('\', '/'))"
    "TEMP_STORAGE_DIR" = $TemporaryRoot
    "KEEP_UPLOADS" = "false"
    "APP_ENV" = "production"
    "OPERATOR_API_ENABLED" = "false"
    "OCR_ENABLED" = "true"
    "PHASE6A_ENABLED" = "true"
    "SEGMENTATION_ENABLED" = "true"
    "SEGMENTATION_MODEL_DIR" = (Join-Path $ProjectRoot ".local\models\atlaslens-segformer-b2-v4")
    "SEGMENTATION_DEVICE" = "auto"
    "PHASE6B_ENABLED" = "true"
    "PHASE6C_ENABLED" = "true"
    "PHASE6C_PIPELINE_VERSION" = "phase6c-v1"
    "PHASE6C_FUSION_CONFIG" = (Join-Path $ProjectRoot "config\reranking\phase6c-v1.json")
    "PHASE6C_OCR_CONFIG" = (Join-Path $ProjectRoot "config\ocr\phase6c-v1.json")
    "GEOCLIP_HIERARCHICAL_ENABLED" = "true"
    "GEOCLIP_GLOBAL_GRID_ENABLED" = "true"
    "GEOCLIP_TURKIYE_REFINEMENT_ENABLED" = "true"
    "MEGALOC_ENABLED" = "true"
    "MEGALOC_WORKER_ENABLED" = "true"
    "MEGALOC_WORKER_HOST" = "127.0.0.1"
    "MEGALOC_WORKER_PORT" = "8794"
    "MEGALOC_DEVICE" = "auto"
    "REFERENCE_INDEX_ENABLED" = "true"
    "REFERENCE_INDEX_PATH" = $ResolvedIndex
    "G3_ENABLED" = "false"
    "OPENAI_GEO_ENABLED" = "false"
    "ENABLE_MOCK_INFERENCE" = "false"
}
$PreviousEnvironment = @{}
$ApiProcess = $null
try {
    foreach ($Entry in $EnvironmentOverrides.GetEnumerator()) {
        $PreviousEnvironment[$Entry.Key] = [Environment]::GetEnvironmentVariable(
            $Entry.Key,
            [EnvironmentVariableTarget]::Process
        )
        [Environment]::SetEnvironmentVariable(
            $Entry.Key,
            $Entry.Value,
            [EnvironmentVariableTarget]::Process
        )
    }
    $LogPath = Join-Path $RunRoot "api.stdout.log"
    $ErrorLogPath = Join-Path $RunRoot "api.stderr.log"
    $ApiProcess = Start-Process `
        -FilePath $Python `
        -ArgumentList @(
            "-m", "uvicorn", "atlaslens_api.main:app",
            "--host", "127.0.0.1", "--port", $Port.ToString(),
            "--log-level", "warning"
        ) `
        -WorkingDirectory (Join-Path $ProjectRoot "services\api") `
        -RedirectStandardOutput $LogPath `
        -RedirectStandardError $ErrorLogPath `
        -WindowStyle Hidden `
        -PassThru

    & $Python (Join-Path $PSScriptRoot "smoke_phase6c_http.py") `
        --base-url "http://127.0.0.1:$Port" `
        --image $ResolvedImage `
        --output $ResolvedOutput
    if ($LASTEXITCODE -ne 0) { throw "The Phase 6C HTTP smoke failed." }
    & $Python (Join-Path $PSScriptRoot "score_phase6c_holdout.py") `
        --api-base-url "http://127.0.0.1:$Port" `
        --prediction-report $ResolvedOutput `
        --holdout-manifest (Join-Path $ProjectRoot ".local\evaluation\turkey_holdout.json") `
        --holdout-id "user-holdout-001" `
        --catalogue (Join-Path $ProjectRoot "assets\geolocation\coordinate_catalogue_v1.json") `
        --output $ResolvedScoreOutput
    if ($LASTEXITCODE -ne 0) { throw "The post-prediction Phase 6C holdout scoring failed." }
}
finally {
    if ($null -ne $ApiProcess -and -not $ApiProcess.HasExited) {
        Stop-Process -Id $ApiProcess.Id
        $ApiProcess.WaitForExit(10000)
    }
    foreach ($Entry in $PreviousEnvironment.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable(
            $Entry.Key,
            $Entry.Value,
            [EnvironmentVariableTarget]::Process
        )
    }
}
