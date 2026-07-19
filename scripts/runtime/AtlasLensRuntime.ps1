Set-StrictMode -Version Latest

function ConvertTo-AtlasLensPowerShellLiteral {
    param([Parameter(Mandatory)][string]$Value)

    return "'" + $Value.Replace("'", "''") + "'"
}

function Get-AtlasLensRuntimeConfiguration {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$RepositoryRoot,
        [AllowEmptyString()][string]$RuntimeRoot = ""
    )

    $resolvedRepositoryRoot = (Resolve-Path -LiteralPath $RepositoryRoot -ErrorAction Stop).Path
    $apiDirectory = Join-Path $resolvedRepositoryRoot "services\api"
    $webDirectory = Join-Path $resolvedRepositoryRoot "apps\web"
    $externalRuntime = -not [string]::IsNullOrWhiteSpace($RuntimeRoot)

    if ($externalRuntime) {
        $runtimeCandidate = if ([IO.Path]::IsPathRooted($RuntimeRoot)) {
            $RuntimeRoot
        }
        else {
            Join-Path $resolvedRepositoryRoot $RuntimeRoot
        }
        $resolvedRuntimeRoot = [IO.Path]::GetFullPath($runtimeCandidate)
        $apiEnvironment = Join-Path $resolvedRuntimeRoot "api-venv"
        $uvCache = Join-Path $resolvedRuntimeRoot "uv-cache"
        $npmCache = Join-Path $resolvedRuntimeRoot "npm-cache"
    }
    else {
        $resolvedRuntimeRoot = $null
        $apiEnvironment = Join-Path $apiDirectory ".venv"
        $uvCache = $null
        $npmCache = $null
    }

    return [pscustomobject]@{
        RepositoryRoot = $resolvedRepositoryRoot
        ApiDirectory = $apiDirectory
        ApiSourceDirectory = Join-Path $apiDirectory "src"
        WebDirectory = $webDirectory
        RuntimeRoot = $resolvedRuntimeRoot
        ExternalRuntime = $externalRuntime
        ApiEnvironment = $apiEnvironment
        ApiPython = Join-Path $apiEnvironment "Scripts\python.exe"
        UvCache = $uvCache
        NpmCache = $npmCache
    }
}

function Get-AtlasLensApiRestoreCommand {
    param([Parameter(Mandatory)][object]$Configuration)

    $parts = @()
    if ($Configuration.ExternalRuntime) {
        $parts += '$env:UV_PROJECT_ENVIRONMENT = ' + (
            ConvertTo-AtlasLensPowerShellLiteral $Configuration.ApiEnvironment
        )
        $parts += '$env:UV_CACHE_DIR = ' + (
            ConvertTo-AtlasLensPowerShellLiteral $Configuration.UvCache
        )
    }
    $parts += 'uv sync --project ' + (
        ConvertTo-AtlasLensPowerShellLiteral $Configuration.ApiDirectory
    ) + ' --frozen --all-groups'
    return $parts -join "; "
}

function Get-AtlasLensWebRestoreCommand {
    param([Parameter(Mandatory)][object]$Configuration)

    $command = 'npm.cmd --prefix ' + (
        ConvertTo-AtlasLensPowerShellLiteral $Configuration.WebDirectory
    ) + ' ci --no-audit --no-fund'
    if ($Configuration.ExternalRuntime) {
        $command += ' --cache ' + (ConvertTo-AtlasLensPowerShellLiteral $Configuration.NpmCache)
    }
    return $command
}

function Set-AtlasLensProcessVariable {
    param(
        [Parameter(Mandatory)][hashtable]$Snapshot,
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Value
    )

    if (-not $Snapshot.ContainsKey($Name)) {
        $existing = Get-Item -LiteralPath "Env:$Name" -ErrorAction SilentlyContinue
        $Snapshot[$Name] = [pscustomobject]@{
            Exists = $null -ne $existing
            Value = if ($null -eq $existing) { $null } else { $existing.Value }
        }
    }
    Set-Item -LiteralPath "Env:$Name" -Value $Value
}

function Set-AtlasLensRuntimeEnvironment {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][object]$Configuration,
        [switch]$IncludeCaches,
        [switch]$IncludeOfflineModelGuards
    )

    $snapshot = @{}
    $pythonPath = $Configuration.ApiSourceDirectory
    $existingPythonPath = Get-Item -LiteralPath "Env:PYTHONPATH" -ErrorAction SilentlyContinue
    if ($null -ne $existingPythonPath -and -not [string]::IsNullOrWhiteSpace($existingPythonPath.Value)) {
        $pythonPath += [IO.Path]::PathSeparator + $existingPythonPath.Value
    }
    Set-AtlasLensProcessVariable -Snapshot $snapshot -Name "PYTHONPATH" -Value $pythonPath

    if ($IncludeCaches -and $Configuration.ExternalRuntime) {
        Set-AtlasLensProcessVariable -Snapshot $snapshot -Name "UV_PROJECT_ENVIRONMENT" -Value $Configuration.ApiEnvironment
        Set-AtlasLensProcessVariable -Snapshot $snapshot -Name "UV_CACHE_DIR" -Value $Configuration.UvCache
        Set-AtlasLensProcessVariable -Snapshot $snapshot -Name "npm_config_cache" -Value $Configuration.NpmCache
    }

    if ($IncludeOfflineModelGuards) {
        foreach ($entry in @{
            HF_DATASETS_OFFLINE = "1"
            HF_HUB_DISABLE_TELEMETRY = "1"
            HF_HUB_OFFLINE = "1"
            TRANSFORMERS_OFFLINE = "1"
        }.GetEnumerator()) {
            Set-AtlasLensProcessVariable -Snapshot $snapshot -Name $entry.Key -Value $entry.Value
        }
    }

    return $snapshot
}

function Restore-AtlasLensRuntimeEnvironment {
    param([Parameter(Mandatory)][hashtable]$Snapshot)

    foreach ($entry in $Snapshot.GetEnumerator()) {
        if ($entry.Value.Exists) {
            Set-Item -LiteralPath "Env:$($entry.Key)" -Value $entry.Value.Value
        }
        else {
            Remove-Item -LiteralPath "Env:$($entry.Key)" -ErrorAction SilentlyContinue
        }
    }
}

function Assert-AtlasLensApiRuntime {
    param([Parameter(Mandatory)][object]$Configuration)

    $restoreCommand = Get-AtlasLensApiRestoreCommand $Configuration
    if (-not (Test-Path -LiteralPath $Configuration.ApiPython -PathType Leaf)) {
        throw "AtlasLens API environment is missing. Restore the locked environment with: $restoreCommand"
    }

    & $Configuration.ApiPython -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw "AtlasLens API requires Python 3.12. Restore the locked environment with: $restoreCommand"
    }

    & $Configuration.ApiPython -c "import alembic, atlaslens_api, uvicorn"
    if ($LASTEXITCODE -ne 0) {
        throw "AtlasLens API dependencies or source imports are unavailable. Restore the locked environment with: $restoreCommand"
    }
}

function Get-AtlasLensFreeSpaceGiB {
    param([Parameter(Mandatory)][string]$DriveName)

    $drive = Get-PSDrive -PSProvider FileSystem -Name $DriveName -ErrorAction Stop
    return [double]($drive.Free / 1GB)
}

function Assert-AtlasLensDiskBudget {
    foreach ($requirement in @(
        [pscustomobject]@{ Drive = "D"; MinimumGiB = 8.0 },
        [pscustomobject]@{ Drive = "C"; MinimumGiB = 35.0 }
    )) {
        $freeGiB = Get-AtlasLensFreeSpaceGiB $requirement.Drive
        if ($freeGiB -lt $requirement.MinimumGiB) {
            $formatted = $freeGiB.ToString("0.00", [Globalization.CultureInfo]::InvariantCulture)
            throw "Disk safety gate failed: drive $($requirement.Drive) has $formatted GiB free; at least $($requirement.MinimumGiB) GiB is required."
        }
    }
}
