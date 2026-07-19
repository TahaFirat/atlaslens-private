[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$webRoot = Join-Path $repoRoot "apps\web"
$startScript = [IO.Path]::GetFullPath((Join-Path $repoRoot "scripts\start-investor-demo.ps1"))
$stopScript = [IO.Path]::GetFullPath((Join-Path $repoRoot "scripts\stop-investor-demo.ps1"))
$runtimeRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot ".local\run\investor-demo"))
$workerMetadata = [IO.Path]::GetFullPath((Join-Path $repoRoot ".local\run\phase6c-workers\megaloc.json"))
$externalSpec = "e2e/investor-demo-external.spec.ts"
$expectedPorts = @(8794, 8000, 5173)
$powershell = (Get-Command "powershell.exe" -ErrorAction Stop).Source
$cmd = (Get-Command "cmd.exe" -ErrorAction Stop).Source
$npm = (Get-Command "npm.cmd" -ErrorAction Stop).Source
$outsideWorkingDirectory = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$visualRoot = [IO.Path]::GetFullPath(
    (Join-Path $outsideWorkingDirectory ("atlaslens-phase3e-visual-" + [Guid]::NewGuid().ToString("N")))
)
$environmentNames = @(
    "ATLASLENS_E2E_EXTERNAL",
    "ATLASLENS_E2E_BASE_URL",
    "ATLASLENS_INVESTOR_DEMO_E2E",
    "ATLASLENS_INVESTOR_DEMO_URL",
    "ATLASLENS_PHASE3E_SCREENSHOT_DIR"
)
$environmentSnapshot = @{}
$failures = [Collections.Generic.List[string]]::new()
$startupAttempted = $false
$launcherStdoutPath = $null
$launcherStderrPath = $null

foreach ($required in @($startScript, $stopScript, (Join-Path $webRoot $externalSpec))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required investor-demo E2E component is missing: $required"
    }
}

$repoPrefix = $repoRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
if (
    $outsideWorkingDirectory.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase) -or
    [string]::Equals($outsideWorkingDirectory, $repoRoot, [StringComparison]::OrdinalIgnoreCase)
) {
    throw "The investor-demo launcher working directory must be outside the repository."
}

function ConvertFrom-QueryString([Uri]$Uri) {
    $values = @{}
    foreach ($part in $Uri.Query.TrimStart("?").Split("&", [StringSplitOptions]::RemoveEmptyEntries)) {
        $pieces = $part.Split(@("="), 2, [StringSplitOptions]::None)
        $name = [Uri]::UnescapeDataString($pieces[0].Replace("+", " "))
        $value = if ($pieces.Count -eq 2) {
            [Uri]::UnescapeDataString($pieces[1].Replace("+", " "))
        }
        else { "" }
        if ($values.ContainsKey($name)) {
            throw "The printed demo URL contains a duplicate query parameter."
        }
        $values[$name] = $value
    }
    return $values
}

function Get-OfficialDemoUrl([string[]]$OutputLines) {
    $demoMatches = @(
        foreach ($line in $OutputLines) {
            $match = [regex]::Match($line, "^Demo: (?<url>\S+)$")
            if ($match.Success) { $match.Groups["url"].Value }
        }
    )
    if ($demoMatches.Count -ne 1) {
        throw "The official launcher did not print exactly one case-bearing Demo URL."
    }
    try { $uri = [Uri]$demoMatches[0] }
    catch { throw "The official launcher printed an invalid Demo URL." }
    $query = ConvertFrom-QueryString $uri
    $allowedNames = @("caseId", "demo", "lang")
    if (
        -not $uri.IsAbsoluteUri -or
        $uri.Scheme -ne "http" -or
        $uri.Host -ne "127.0.0.1" -or
        $uri.Port -ne 5173 -or
        $uri.AbsolutePath -ne "/" -or
        -not [string]::IsNullOrEmpty($uri.Fragment) -or
        $query.Count -ne $allowedNames.Count -or
        @($query.Keys | Where-Object { $_ -notin $allowedNames }).Count -gt 0 -or
        $query["demo"] -ne "investor" -or
        $query["lang"] -ne "tr"
    ) {
        throw "The official launcher Demo URL violates the loopback route contract."
    }
    $caseId = [Guid]::Empty
    if (-not [Guid]::TryParseExact([string]$query["caseId"], "D", [ref]$caseId)) {
        throw "The official launcher Demo URL does not contain a canonical caseId."
    }
    return $uri
}

function Assert-InvestorDemoCleanup {
    $listeners = @(
        Get-NetTCPConnection -State Listen -ErrorAction Stop |
            Where-Object { [int]$_.LocalPort -in $expectedPorts }
    )
    if ($listeners.Count -gt 0) {
        $ports = @($listeners | Select-Object -ExpandProperty LocalPort -Unique | Sort-Object) -join ", "
        throw "Investor-demo cleanup left listeners on expected ports: $ports"
    }
    if (Test-Path -LiteralPath $runtimeRoot) {
        throw "Investor-demo cleanup retained its private runtime directory."
    }
    if (Test-Path -LiteralPath $workerMetadata) {
        throw "Investor-demo cleanup retained trusted MegaLoc PID metadata."
    }
}

foreach ($name in $environmentNames) {
    $environmentSnapshot[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

try {
    try {
        $startupAttempted = $true
        $launcherStdoutPath = [IO.Path]::GetTempFileName()
        $launcherStderrPath = [IO.Path]::GetTempFileName()
        $launcherInfo = [Diagnostics.ProcessStartInfo]::new()
        $launcherInfo.FileName = $cmd
        $launcherInfo.Arguments = (
            "/d /s /c `"`"$powershell`" -NoProfile -ExecutionPolicy Bypass " +
            "-File `"$startScript`" 1>`"$launcherStdoutPath`" " +
            "2>`"$launcherStderrPath`"`""
        )
        $launcherInfo.WorkingDirectory = $outsideWorkingDirectory
        $launcherInfo.UseShellExecute = $false
        $launcherInfo.CreateNoWindow = $true
        $launcherProcess = [Diagnostics.Process]::new()
        $launcherProcess.StartInfo = $launcherInfo
        try {
            if (-not $launcherProcess.Start()) {
                throw "The official investor-demo launcher could not be started."
            }
            $launcherProcess.WaitForExit()
            $startExitCode = $launcherProcess.ExitCode
        }
        finally { $launcherProcess.Dispose() }
        $startupOutput = @(Get-Content -LiteralPath $launcherStdoutPath)
        $startupErrors = @(Get-Content -LiteralPath $launcherStderrPath)
        foreach ($line in $startupOutput) { Write-Host ([string]$line) }
        foreach ($line in $startupErrors) { Write-Host ([string]$line) }
        if ($startExitCode -ne 0) {
            throw "The official investor-demo launcher failed with code $startExitCode."
        }

        $demoUrl = Get-OfficialDemoUrl $startupOutput
        $baseUrl = $demoUrl.GetLeftPart([UriPartial]::Authority)
        [Environment]::SetEnvironmentVariable("ATLASLENS_E2E_EXTERNAL", "1", "Process")
        [Environment]::SetEnvironmentVariable("ATLASLENS_E2E_BASE_URL", $baseUrl, "Process")
        [Environment]::SetEnvironmentVariable("ATLASLENS_INVESTOR_DEMO_E2E", "1", "Process")
        [Environment]::SetEnvironmentVariable("ATLASLENS_INVESTOR_DEMO_URL", $demoUrl.AbsoluteUri, "Process")
        New-Item -ItemType Directory -Path $visualRoot -ErrorAction Stop | Out-Null
        [Environment]::SetEnvironmentVariable(
            "ATLASLENS_PHASE3E_SCREENSHOT_DIR",
            $visualRoot,
            "Process"
        )
        Write-Host "Visuals: $visualRoot"

        Push-Location -LiteralPath $webRoot
        try {
            & $npm run test:e2e -- --project=chromium $externalSpec --reporter=list
            $playwrightExitCode = $LASTEXITCODE
        }
        finally { Pop-Location }
        if ($playwrightExitCode -ne 0) {
            throw "The external investor-demo Playwright spec failed with code $playwrightExitCode."
        }
    }
    catch {
        $failures.Add($_.Exception.Message)
    }
}
finally {
    if ($startupAttempted) {
        try {
            Push-Location -LiteralPath $outsideWorkingDirectory
            try {
                & $powershell -NoProfile -ExecutionPolicy Bypass -File $stopScript 2>&1 |
                    ForEach-Object { Write-Host ([string]$_) }
                $stopExitCode = $LASTEXITCODE
            }
            finally { Pop-Location }
            if ($stopExitCode -ne 0) {
                throw "The official investor-demo stop script failed with code $stopExitCode."
            }
            Assert-InvestorDemoCleanup
        }
        catch {
            $failures.Add($_.Exception.Message)
        }
    }
    foreach ($name in $environmentNames) {
        [Environment]::SetEnvironmentVariable($name, $environmentSnapshot[$name], "Process")
    }
    foreach ($capturePath in @($launcherStdoutPath, $launcherStderrPath)) {
        if (
            -not [string]::IsNullOrWhiteSpace([string]$capturePath) -and
            (Test-Path -LiteralPath $capturePath -PathType Leaf)
        ) {
            Remove-Item -LiteralPath $capturePath -Force
        }
    }
}

if ($failures.Count -gt 0) {
    throw ($failures -join [Environment]::NewLine)
}

Write-Host "External investor-demo Playwright gate passed and official cleanup was verified."
