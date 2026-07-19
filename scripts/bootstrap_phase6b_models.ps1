[CmdletBinding()]
param(
    [ValidateSet("osv5m", "plonk", "paddleocr", "rapidocr")]
    [string[]]$Models = @("osv5m", "plonk", "paddleocr", "rapidocr"),
    [switch]$VerifyOnly,
    [switch]$Offline
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ManifestPath = Join-Path $ProjectRoot "config\external-models.lock.json"
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
$LocalRoot = Join-Path $ProjectRoot ".local"
$VendorRoot = Join-Path $LocalRoot "vendor"
$WorkerRoot = Join-Path $LocalRoot "workers"
$WeightRoot = Join-Path $LocalRoot "models\phase6b"
$env:UV_CACHE_DIR = Join-Path $LocalRoot "uv-cache"
New-Item -ItemType Directory -Force -Path $env:UV_CACHE_DIR | Out-Null

function Invoke-Checked {
    param([Parameter(Mandatory)][string]$FilePath, [Parameter(Mandatory)][string[]]$Arguments)
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $FilePath $($Arguments -join ' ')"
    }
}

function Get-ModelLock {
    param([Parameter(Mandatory)][string]$Name)
    $entry = $Manifest.models | Where-Object { $_.name -eq $Name } | Select-Object -First 1
    if ($null -eq $entry) { throw "No manifest entry exists for $Name." }
    return $entry
}

function Get-PythonLauncher {
    param([Parameter(Mandatory)][string]$Version)
    & py "-$Version" -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Python $Version is required for this isolated worker. Install that exact minor version, then rerun. The main API environment will not be changed."
    }
}

function Ensure-PinnedCheckout {
    param([Parameter(Mandatory)]$Lock, [switch]$NoCheckout)
    $target = Join-Path $VendorRoot $Lock.name
    if (-not (Test-Path -LiteralPath (Join-Path $target ".git"))) {
        Invoke-Checked git @("clone", "--filter=blob:none", "--no-checkout", $Lock.source_repository, $target)
    }
    $gitSafe = ([IO.Path]::GetFullPath($target)).Replace("\", "/")
    & git -c "safe.directory=$gitSafe" -C $target cat-file -e "$($Lock.source_revision)^{commit}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Invoke-Checked git @("-c", "safe.directory=$gitSafe", "-C", $target, "fetch", "--depth=1", "origin", $Lock.source_revision)
    }
    # Materialize only sources imported by a worker. Package-only providers need
    # the pinned commit object for provenance but do not need a network-dependent
    # checkout of unused source blobs.
    if (-not $NoCheckout) {
        Invoke-Checked git @("-c", "safe.directory=$gitSafe", "-C", $target, "checkout", "--detach", "--force", $Lock.source_revision)
    }
    $revisionToVerify = if ($NoCheckout) { "$($Lock.source_revision)^{commit}" } else { "HEAD" }
    $verified = (& git -c "safe.directory=$gitSafe" -C $target rev-parse $revisionToVerify).Trim()
    if ($verified -ne $Lock.source_revision) {
        throw "$($Lock.name) source revision verification failed. Expected $($Lock.source_revision), got $verified."
    }
    return $target
}

function Ensure-WorkerEnvironment {
    param([Parameter(Mandatory)][string]$Name, [Parameter(Mandatory)][string]$PythonVersion)
    $venv = Join-Path (Join-Path $WorkerRoot $Name) ".venv"
    $python = Join-Path $venv "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python)) {
        Get-PythonLauncher $PythonVersion | Out-Null
        New-Item -ItemType Directory -Force -Path (Split-Path $venv) | Out-Null
        Invoke-Checked uv @("venv", "--python", $PythonVersion, $venv)
    }
    $actualVersion = (& $python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
    if ($LASTEXITCODE -ne 0 -or $actualVersion -ne $PythonVersion) {
        throw "$Name worker Python mismatch. Expected $PythonVersion, got $actualVersion."
    }
    return $python
}

function Install-HuggingFaceClient {
    param([Parameter(Mandatory)][string]$Python)
    $version = $Manifest.tooling.huggingface_hub
    if ([string]::IsNullOrWhiteSpace($version)) {
        throw "The external-model lock is missing tooling.huggingface_hub."
    }
    # Keep the worker CLI compatible with its Transformers stack. Pinning the API
    # environment's older Hub client here breaks current isolated workers.
    Invoke-Checked uv @("pip", "install", "--python", $Python, "huggingface-hub==$version")
}

function Download-HuggingFaceSnapshot {
    param(
        [Parameter(Mandatory)][string]$Python,
        [Parameter(Mandatory)][string]$Repository,
        [Parameter(Mandatory)][string]$Revision,
        [Parameter(Mandatory)][string]$Destination
    )
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    # huggingface-hub changed both console-script and CLI module names between
    # reviewed releases. The public Python API is stable and keeps arguments out
    # of generated source by reading them from sys.argv.
    $downloadCode = "from huggingface_hub import snapshot_download; import sys; snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])"
    Invoke-Checked $Python @(
        "-c", $downloadCode,
        $Repository, $Revision, $Destination
    )
}

function Install-Osv5m {
    $lock = Get-ModelLock "osv5m"
    $source = Ensure-PinnedCheckout $lock
    $python = Ensure-WorkerEnvironment "osv5m" $lock.python
    $requirements = Join-Path $source "requirements.txt"
    if (-not (Test-Path -LiteralPath $requirements -PathType Leaf)) {
        throw "The pinned OSV-5M checkout is incomplete: requirements.txt is missing."
    }
    # The official OSV-5M repository is source-only (no setup.py/pyproject.toml).
    # Its documented inference runs with the repository as the Python working tree.
    Invoke-Checked uv @("pip", "install", "--python", $python, "-r", $requirements)
    Invoke-Checked uv @(
        "pip", "install", "--python", $python,
        "torchvision==$($Manifest.tooling.osv5m_torchvision)"
    )
    Install-HuggingFaceClient $python
    Invoke-Checked uv @("pip", "check", "--python", $python)
    Download-HuggingFaceSnapshot $python $lock.model_id $lock.model_revision (Join-Path $WeightRoot "osv5m")
}

function Install-Plonk {
    $lock = Get-ModelLock "plonk"
    Ensure-PinnedCheckout $lock -NoCheckout | Out-Null
    $python = Ensure-WorkerEnvironment "plonk" $lock.python
    Invoke-Checked uv @("pip", "install", "--python", $python, "diff-plonk==0.4")
    Install-HuggingFaceClient $python
    Invoke-Checked uv @("pip", "check", "--python", $python)
    foreach ($variant in $lock.variants) {
        Download-HuggingFaceSnapshot $python $variant.model_id $variant.revision (Join-Path (Join-Path $WeightRoot "plonk") $variant.name)
    }
    $auxiliary = $lock.auxiliary
    if ($null -eq $auxiliary) { throw "The PLONK lock is missing its DINOv2 auxiliary model." }
    $dinoLock = [pscustomobject]@{
        name = $auxiliary.name
        source_repository = $auxiliary.source_repository
        source_revision = $auxiliary.source_revision
    }
    $dinoSource = Ensure-PinnedCheckout $dinoLock
    $previousTorchHome = $env:TORCH_HOME
    try {
        $env:TORCH_HOME = Join-Path (Join-Path $WeightRoot "plonk") "torch"
        $dinoCode = "import sys,torch; torch.hub.load(sys.argv[1],sys.argv[2],source='local',pretrained=True,verbose=False)"
        Invoke-Checked $python @("-c", $dinoCode, $dinoSource, $auxiliary.model_id)
    }
    finally {
        $env:TORCH_HOME = $previousTorchHome
    }
    $dinoWeight = Join-Path (Join-Path (Join-Path (Join-Path $WeightRoot "plonk") "torch") "hub\checkpoints") "dinov2_vitl14_reg4_pretrain.pth"
    if (-not (Test-Path -LiteralPath $dinoWeight -PathType Leaf)) {
        throw "The pinned DINOv2 inference weight is missing after bootstrap."
    }
    $dinoHash = (Get-FileHash -LiteralPath $dinoWeight -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($dinoHash -ne $auxiliary.weight_sha256) {
        throw "The pinned DINOv2 inference weight checksum is invalid."
    }
}

function Install-PaddleOcr {
    $lock = Get-ModelLock "paddleocr"
    Ensure-PinnedCheckout $lock -NoCheckout | Out-Null
    $python = Ensure-WorkerEnvironment "paddleocr" $lock.python
    Invoke-Checked uv @("pip", "install", "--python", $python, "paddlepaddle==3.3.1", "paddleocr==3.7.0")
    $paddleRoot = Join-Path $WeightRoot "paddleocr"
    $detector = Join-Path $paddleRoot "det"
    $recognizer = Join-Path $paddleRoot "rec\latin_PP-OCRv5_mobile_rec"
    $privateHome = Join-Path $paddleRoot "bootstrap-home"
    New-Item -ItemType Directory -Force -Path $privateHome, (Join-Path $privateHome ".cache\paddle\dataset") | Out-Null
    $oldUserProfile, $oldHome = $env:USERPROFILE, $env:HOME
    try {
        $env:USERPROFILE = $privateHome
        $env:HOME = $privateHome
        $env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"
        $downloadCode = "from paddleocr import PaddleOCR; p=PaddleOCR(text_detection_model_name='PP-OCRv5_server_det',text_recognition_model_name='latin_PP-OCRv5_mobile_rec',use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False,device='cpu',enable_mkldnn=False); p.close()"
        if (-not ((Test-Path (Join-Path $detector "inference.pdiparams")) -and (Test-Path (Join-Path $recognizer "inference.pdiparams")))) {
            Invoke-Checked $python @("-c", $downloadCode)
            $cacheRoot = Join-Path $privateHome ".paddlex\official_models"
            if (-not (Test-Path $detector)) {
                Copy-Item -Recurse -LiteralPath (Join-Path $cacheRoot "PP-OCRv5_server_det") -Destination $detector
            }
            if (-not (Test-Path $recognizer)) {
                New-Item -ItemType Directory -Force -Path (Split-Path $recognizer) | Out-Null
                Copy-Item -Recurse -LiteralPath (Join-Path $cacheRoot "latin_PP-OCRv5_mobile_rec") -Destination $recognizer
            }
        }
        $verifyCode = "from paddleocr import PaddleOCR; import sys; p=PaddleOCR(text_detection_model_name='PP-OCRv5_server_det',text_detection_model_dir=sys.argv[1],text_recognition_model_name='latin_PP-OCRv5_mobile_rec',text_recognition_model_dir=sys.argv[2],use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False,device='cpu',enable_mkldnn=False); p.close()"
        Invoke-Checked $python @("-c", $verifyCode, $detector, $recognizer)
    }
    finally {
        $env:USERPROFILE, $env:HOME = $oldUserProfile, $oldHome
        Remove-Item Env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK -ErrorAction SilentlyContinue
    }
    Invoke-Checked uv @("pip", "check", "--python", $python)
}

function Install-RapidOcr {
    $apiPython = Join-Path $ProjectRoot "services\api\.venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $apiPython -PathType Leaf)) {
        throw "The API environment is required before installing the existing RapidOCR fallback."
    }
    Invoke-Checked uv @("pip", "install", "--python", $apiPython, "rapidocr==3.9.1", "onnxruntime==1.27.0")
    Invoke-Checked uv @("pip", "check", "--python", $apiPython)
    Invoke-Checked $apiPython @("-m", "atlaslens_api.cli", "models", "install", "rapidocr")
    Invoke-Checked $apiPython @("-m", "atlaslens_api.cli", "models", "verify", "rapidocr")
    $rapidRoot = if ($env:RAPIDOCR_MODEL_ROOT) {
        $env:RAPIDOCR_MODEL_ROOT
    }
    elseif ($env:LOCALAPPDATA) {
        Join-Path $env:LOCALAPPDATA "AtlasLens\models\rapidocr-3.9.1"
    }
    else {
        Join-Path $HOME ".atlaslens\models\rapidocr-3.9.1"
    }
    Invoke-Checked $apiPython @(
        (Join-Path $PSScriptRoot "verify_rapidocr_runtime.py"),
        "--project-root", $ProjectRoot,
        "--model-root", $rapidRoot
    )
}

function Invoke-Verification {
    $apiPython = Join-Path $ProjectRoot "services\api\.venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $apiPython)) { $apiPython = "python" }
    Invoke-Checked $apiPython @((Join-Path $PSScriptRoot "verify_phase6b_models.py"), "--project-root", $ProjectRoot)
}

if ($VerifyOnly -or $Offline) {
    if ($Offline) {
        Write-Host "Offline mode: no clone, package installation, model download, or dataset action will run."
    }
    Invoke-Verification
    exit 0
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "uv is required for isolated, reproducible worker environments." }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "Git is required to verify pinned external source revisions." }

$drive = [System.IO.DriveInfo]::new((Split-Path -Qualifier $ProjectRoot))
$minimumFree = 8GB
if ($drive.AvailableFreeSpace -lt $minimumFree) {
    throw "At least 8 GiB free space is required before preparing Phase 6B model artifacts. Available: $([math]::Round($drive.AvailableFreeSpace / 1GB, 2)) GiB."
}

New-Item -ItemType Directory -Force -Path $VendorRoot, $WorkerRoot, $WeightRoot | Out-Null
Write-Host "Phase 6B policy: pretrained inference weights only; dataset downloads are refused."
Write-Host "Main API PyTorch/CUDA environment (read-only check):"
$apiPython = Join-Path $ProjectRoot "services\api\.venv\Scripts\python.exe"
if (Test-Path -LiteralPath $apiPython) {
    & $apiPython -c "import torch; print('torch=' + torch.__version__ + '; cuda_runtime=' + str(torch.version.cuda) + '; cuda_available=' + str(torch.cuda.is_available()))"
}

foreach ($model in $Models) {
    switch ($model) {
        "osv5m" { Install-Osv5m }
        "plonk" { Install-Plonk }
        "paddleocr" { Install-PaddleOcr }
        "rapidocr" { Install-RapidOcr }
        default { throw "Dataset or unknown component '$model' is not accepted by this weights-only bootstrap." }
    }
}

Invoke-Verification
