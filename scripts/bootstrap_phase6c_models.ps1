[CmdletBinding()]
param(
    [ValidateSet("megaloc")]
    [string[]]$Models = @("megaloc"),
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [switch]$VerifyOnly,
    [switch]$ArtifactsOnly,
    [switch]$Offline
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ManifestPath = Join-Path $ProjectRoot "config\external-models.lock.json"
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
$VendorRoot = Join-Path $ProjectRoot ".local\vendor"
$WorkerRoot = Join-Path $ProjectRoot ".local\workers"
$WeightRoot = Join-Path $ProjectRoot ".local\models\phase6c"
$env:UV_CACHE_DIR = Join-Path $ProjectRoot ".local\uv-cache"

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
    if ($null -eq $entry) { throw "No external-model lock exists for $Name." }
    return $entry
}

function Ensure-PinnedCheckout {
    param([Parameter(Mandatory)]$Lock)
    $target = Join-Path $VendorRoot $Lock.name
    if (-not (Test-Path -LiteralPath (Join-Path $target ".git"))) {
        Invoke-Checked git @("clone", "--filter=blob:none", $Lock.source_repository, $target)
    }
    $safe = ([IO.Path]::GetFullPath($target)).Replace("\", "/")
    & git -c "safe.directory=$safe" -C $target cat-file -e "$($Lock.source_revision)^{commit}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Invoke-Checked git @("-c", "safe.directory=$safe", "-C", $target, "fetch", "--depth=1", "origin", $Lock.source_revision)
    }
    Invoke-Checked git @("-c", "safe.directory=$safe", "-C", $target, "checkout", "--detach", "--force", $Lock.source_revision)
    $actual = (& git -c "safe.directory=$safe" -C $target rev-parse HEAD).Trim()
    if ($actual -ne $Lock.source_revision) {
        throw "$($Lock.name) source revision mismatch."
    }
    if (-not (Test-Path -LiteralPath (Join-Path $target "LICENSE") -PathType Leaf)) {
        throw "$($Lock.name) license file is missing."
    }
    return $target
}

function Ensure-WorkerEnvironment {
    param([Parameter(Mandatory)]$Lock)
    $venv = Join-Path (Join-Path $WorkerRoot $Lock.name) ".venv"
    $python = Join-Path $venv "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        New-Item -ItemType Directory -Force -Path (Split-Path $venv) | Out-Null
        Invoke-Checked uv @("venv", "--python", $Lock.python, $venv)
    }
    $actual = (& $python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
    if ($actual -ne $Lock.python) { throw "$($Lock.name) worker Python mismatch." }
    return $python
}

function Install-MegaLoc {
    $lock = Get-ModelLock "megaloc"
    Ensure-PinnedCheckout $lock | Out-Null
    $python = Ensure-WorkerEnvironment $lock
    $packages = $lock.packages
    Invoke-Checked uv @(
        "pip", "install", "--python", $python,
        "--index", "https://download.pytorch.org/whl/cu128",
        "torch==$($packages.torch)", "torchvision==$($packages.torchvision)"
    )
    Invoke-Checked uv @(
        "pip", "install", "--python", $python,
        "huggingface-hub==$($packages.huggingface_hub)",
        "safetensors==$($packages.safetensors)", "pillow==$($packages.pillow)"
    )
    Invoke-Checked uv @("pip", "check", "--python", $python)

    $modelDir = Join-Path $WeightRoot "megaloc"
    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
    $weight = Join-Path $modelDir $lock.weight_file
    $existingIsPinned = $false
    if (Test-Path -LiteralPath $weight -PathType Leaf) {
        $existingSize = (Get-Item -LiteralPath $weight).Length
        if ($existingSize -eq [long]$lock.weight_size) {
            $existingSha = (Get-FileHash -LiteralPath $weight -Algorithm SHA256).Hash.ToLowerInvariant()
            $existingIsPinned = $existingSha -eq $lock.weight_sha256
        }
    }
    if (-not $existingIsPinned) {
        if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
            throw "curl.exe is required for the pinned checkpoint transfer."
        }
        $repository = [System.Uri]::EscapeDataString([string]$lock.model_id).Replace("%2F", "/")
        $revision = [System.Uri]::EscapeDataString([string]$lock.model_revision)
        $filename = [System.Uri]::EscapeDataString([string]$lock.weight_file)
        $downloadUrl = "https://huggingface.co/$repository/resolve/$revision/$filename?download=true"
        Invoke-Checked curl.exe @(
            "-L", "--fail", "--retry", "3", "--retry-delay", "3",
            "--output", $weight, $downloadUrl
        )
    }
    if (-not (Test-Path -LiteralPath $weight -PathType Leaf)) { throw "MegaLoc weight is missing." }
    if ((Get-Item -LiteralPath $weight).Length -ne [long]$lock.weight_size) {
        throw "MegaLoc weight size mismatch."
    }
    $sha = (Get-FileHash -LiteralPath $weight -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($sha -ne $lock.weight_sha256) { throw "MegaLoc weight checksum mismatch." }
    $receipt = [ordered]@{
        schema_version = "atlaslens-model-receipt-v1"
        provider = "megaloc"
        source_revision = $lock.source_revision
        model_id = $lock.model_id
        model_revision = $lock.model_revision
        weight_file = $lock.weight_file
        weight_size = [long]$lock.weight_size
        weight_sha256 = $lock.weight_sha256
        license = $lock.license
        datasets_downloaded = $false
    }
    $receiptJson = $receipt | ConvertTo-Json -Depth 4
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText((Join-Path $modelDir "receipt.json"), $receiptJson, $utf8NoBom)
}

function Invoke-Verification {
    $lock = Get-ModelLock "megaloc"
    $python = Join-Path (Join-Path (Join-Path $WorkerRoot $lock.name) ".venv") "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "MegaLoc worker environment is not installed."
    }
    $arguments = @(
        (Join-Path $PSScriptRoot "verify_phase6c_models.py"),
        "--project-root", $ProjectRoot,
        "--device", $Device
    )
    if ($ArtifactsOnly) { $arguments += "--artifacts-only" }
    Invoke-Checked $python $arguments
}

if ($VerifyOnly -or $Offline) {
    if ($Offline) {
        Write-Host "Offline mode: no clone, install, model download, or dataset action will run."
    }
    Invoke-Verification
    exit 0
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "uv is required." }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "Git is required." }
$drive = [System.IO.DriveInfo]::new((Split-Path -Qualifier $ProjectRoot))
if ($drive.AvailableFreeSpace -lt 8GB) {
    throw "At least 8 GiB free space is required for the isolated MegaLoc runtime."
}
New-Item -ItemType Directory -Force -Path $VendorRoot, $WorkerRoot, $WeightRoot, $env:UV_CACHE_DIR | Out-Null
Write-Host "Phase 6C policy: pinned inference weights only; dataset and published gallery downloads are refused."
foreach ($model in $Models) {
    switch ($model) {
        "megaloc" { Install-MegaLoc }
        default { throw "Unknown or dataset component '$model' is not accepted." }
    }
}
Invoke-Verification
