# 1-install.ps1 — first-run bootstrap for iMakeAiTeams.
#
# Runs on a clean Windows machine with nothing but PowerShell preinstalled.
# Installs Node.js LTS, Python 3.12, all npm + pip dependencies, and the
# PyInstaller toolchain into a project-local venv. All errors are surfaced
# in plain English with a recovery path.
#
# Hard rules enforced here:
#   * Download fallback chain: Invoke-WebRequest -> System.Net.WebClient ->
#     curl.exe --ssl-no-revoke
#   * pip uses --timeout=1000 --retries=20 --no-cache-dir --only-binary=:all:
#     so wheels never compile at install time
#   * PATH is refreshed mid-session so newly installed binaries are visible
#     without restarting the shell
#   * No silent failures: every failure prints WHAT to do next

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$NODE_MIN_MAJOR = 20
$PYTHON_MIN = [Version]"3.12"

$NODE_MSI_FALLBACK = "https://nodejs.org/dist/v20.18.1/node-v20.18.1-x64.msi"
$PYTHON_EXE_FALLBACK = "https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe"

# ── Pretty printing ──────────────────────────────────────────────────────────

function Write-Step($text)  { Write-Host "==> $text" -ForegroundColor Cyan }
function Write-Ok($text)    { Write-Host "[ok] $text" -ForegroundColor Green }
function Write-Warn2($text) { Write-Host "[warn] $text" -ForegroundColor Yellow }
function Write-Err2($text)  { Write-Host "[error] $text" -ForegroundColor Red }

function Refresh-Path {
    # Combine machine + user PATH so installs done in this session become
    # visible without restarting the shell. Preserve any process-only
    # additions made earlier in this session (some MSIs only touch the
    # process env) by appending the current $env:Path last and de-duping.
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user    = [Environment]::GetEnvironmentVariable("Path", "User")
    $current = $env:Path
    $entries = @()
    foreach ($src in @($machine, $user, $current)) {
        if (-not $src) { continue }
        foreach ($p in $src.Split(';')) {
            if ($p -and -not ($entries -contains $p)) {
                $entries += $p
            }
        }
    }
    $env:Path = ($entries -join ';')
}

function Test-Command($name) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    return ($null -ne $cmd)
}

# ── Download with fallback chain ─────────────────────────────────────────────

function Download-File($Url, $Out) {
    Write-Host "    downloading $Url"
    $tmpDir = Split-Path -Parent $Out
    if (-not (Test-Path $tmpDir)) { New-Item -ItemType Directory -Path $tmpDir | Out-Null }

    # 1) Invoke-WebRequest
    try {
        Invoke-WebRequest -Uri $Url -OutFile $Out -UseBasicParsing -TimeoutSec 600
        if ((Test-Path $Out) -and ((Get-Item $Out).Length -gt 0)) { return $true }
    } catch {
        Write-Warn2 "Invoke-WebRequest failed: $($_.Exception.Message)"
    }

    # 2) System.Net.WebClient
    try {
        $wc = New-Object System.Net.WebClient
        $wc.Headers.Add("User-Agent", "iMakeAiTeams-setup")
        $wc.DownloadFile($Url, $Out)
        if ((Test-Path $Out) -and ((Get-Item $Out).Length -gt 0)) { return $true }
    } catch {
        Write-Warn2 "WebClient failed: $($_.Exception.Message)"
    }

    # 3) curl.exe with --ssl-no-revoke (works around AV/firewall CRL fetches
    #    that block in offline / restricted networks).
    if (Test-Command "curl.exe") {
        try {
            & curl.exe -L --ssl-no-revoke --retry 5 --retry-delay 4 --max-time 600 -o $Out $Url
            if ((Test-Path $Out) -and ((Get-Item $Out).Length -gt 0)) { return $true }
        } catch {
            Write-Warn2 "curl.exe failed: $($_.Exception.Message)"
        }
    }

    return $false
}

# ── Node.js ──────────────────────────────────────────────────────────────────

function Get-NodeMajor {
    if (-not (Test-Command "node")) { return 0 }
    try {
        $version = (& node --version) -replace "^v", ""
        return [int]([Version]$version).Major
    } catch { return 0 }
}

function Install-Node {
    Write-Step "Installing Node.js LTS"

    if (Test-Command "winget") {
        try {
            & winget install --id OpenJS.NodeJS.LTS --silent --accept-package-agreements --accept-source-agreements --source winget
            if ($LASTEXITCODE -eq 0) {
                Refresh-Path
                Write-Ok "Node.js installed via winget"
                return
            }
            Write-Warn2 "winget exit code $LASTEXITCODE; falling back to MSI"
        } catch {
            Write-Warn2 "winget failed: $($_.Exception.Message); falling back to MSI"
        }
    } else {
        Write-Warn2 "winget not available; falling back to MSI"
    }

    $msi = Join-Path $env:TEMP "imakeaiteams-node.msi"
    if (-not (Download-File $NODE_MSI_FALLBACK $msi)) {
        throw "Could not download Node.js installer. Check your internet connection or install Node 20 LTS manually from https://nodejs.org and rerun 1-install.bat."
    }
    Write-Host "    running msiexec /i $msi /quiet /norestart"
    # Pass the args as an array so msiexec sees /i, the MSI path, and the
    # remaining flags as four distinct tokens. The single-string form leaves
    # quoting up to PowerShell + cmd, which has historically broken when the
    # MSI path contains spaces or parentheses.
    $proc = Start-Process msiexec.exe -Wait -PassThru -ArgumentList @("/i", "`"$msi`"", "/quiet", "/norestart")
    if ($proc.ExitCode -ne 0) {
        throw "Node MSI install failed with exit code $($proc.ExitCode). Run the MSI manually: $msi"
    }
    Refresh-Path
    Remove-Item $msi -ErrorAction SilentlyContinue
    Write-Ok "Node.js installed via MSI"
}

# ── Python ───────────────────────────────────────────────────────────────────

function Get-PythonExe {
    foreach ($name in @("python", "python3", "py")) {
        if (Test-Command $name) {
            try {
                $v = & $name -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>$null
                if ($LASTEXITCODE -eq 0 -and $v) {
                    if ([Version]$v -ge $PYTHON_MIN) { return $name }
                }
            } catch { }
        }
    }
    return $null
}

function Install-Python {
    Write-Step "Installing Python 3.12"

    if (Test-Command "winget") {
        try {
            & winget install --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements --source winget
            if ($LASTEXITCODE -eq 0) {
                Refresh-Path
                Write-Ok "Python installed via winget"
                return
            }
            Write-Warn2 "winget exit code $LASTEXITCODE; falling back to python.org installer"
        } catch {
            Write-Warn2 "winget failed: $($_.Exception.Message); falling back to python.org installer"
        }
    }

    $exe = Join-Path $env:TEMP "imakeaiteams-python.exe"
    if (-not (Download-File $PYTHON_EXE_FALLBACK $exe)) {
        throw 'Could not download Python installer. Install Python 3.12+ manually from https://www.python.org/downloads/ — be sure to tick "Add Python to PATH" — and rerun 1-install.bat.'
    }
    Write-Host "    running $exe /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1"
    $proc = Start-Process $exe -Wait -PassThru -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_test=0"
    if ($proc.ExitCode -ne 0) {
        throw "Python installer failed with exit code $($proc.ExitCode). Run it manually: $exe"
    }
    Refresh-Path
    Remove-Item $exe -ErrorAction SilentlyContinue
    Write-Ok "Python installed"
}

# ── Verification helpers ─────────────────────────────────────────────────────

function Assert-Tool($name, $hint) {
    if (-not (Test-Command $name)) {
        throw "$name is not on PATH after install. $hint"
    }
}

# ── Main ─────────────────────────────────────────────────────────────────────

try {
    Write-Step "iMakeAiTeams setup — verifying prerequisites"
    Refresh-Path

    # Node
    if ((Get-NodeMajor) -lt $NODE_MIN_MAJOR) {
        Install-Node
        if ((Get-NodeMajor) -lt $NODE_MIN_MAJOR) {
            throw "Node $NODE_MIN_MAJOR+ is still missing. Restart your terminal and rerun 1-install.bat."
        }
    } else {
        Write-Ok "Node $(& node --version) detected"
    }
    Assert-Tool "npm" "Reinstall Node.js or fix PATH."

    # Python
    $py = Get-PythonExe
    if (-not $py) {
        Install-Python
        $py = Get-PythonExe
        if (-not $py) {
            throw "Python 3.12+ is still missing. Restart your terminal and rerun 1-install.bat."
        }
    } else {
        $pyv = & $py -c 'import sys; print(".".join(map(str, sys.version_info[:3])))'
        Write-Ok "Python $pyv detected at $py"
    }

    # ── npm install ──────────────────────────────────────────────────────────
    Write-Step 'Installing npm dependencies — this can take a few minutes'
    Push-Location $ProjectRoot
    try {
        & npm install --no-fund --no-audit --loglevel=error
        if ($LASTEXITCODE -ne 0) { throw "npm install failed with exit code $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
    Write-Ok "npm dependencies installed"

    # ── Python venv ──────────────────────────────────────────────────────────
    Write-Step "Creating Python virtualenv at backend/.venv"
    $venvDir = Join-Path $ProjectRoot "backend\.venv"
    if (-not (Test-Path $venvDir)) {
        & $py -m venv "$venvDir"
        if ($LASTEXITCODE -ne 0) { throw "python -m venv failed" }
    }
    $venvPython = Join-Path $venvDir "Scripts\python.exe"
    $venvPip    = Join-Path $venvDir "Scripts\pip.exe"
    if (-not (Test-Path $venvPython)) {
        throw "venv creation did not produce $venvPython. Delete backend\.venv and rerun."
    }

    Write-Step "Upgrading pip + wheel inside the venv"
    & $venvPython -m pip install --timeout=1000 --retries=20 --no-cache-dir --only-binary=":all:" --upgrade pip wheel setuptools
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

    Write-Step "Installing sidecar dependencies"
    $reqs = Join-Path $ProjectRoot "backend\requirements.txt"
    & $venvPython -m pip install --timeout=1000 --retries=20 --no-cache-dir --only-binary=":all:" -r "$reqs"
    if ($LASTEXITCODE -ne 0) { throw "pip install -r backend\requirements.txt failed" }

    Write-Step 'Installing PyInstaller — bundled in dev so 3-build-installer.bat can run without re-installing'
    & $venvPython -m pip install --timeout=1000 --retries=20 --no-cache-dir --only-binary=":all:" "pyinstaller==6.11.1"
    if ($LASTEXITCODE -ne 0) { throw "pip install pyinstaller failed" }

    # ── Smoke test ───────────────────────────────────────────────────────────
    Write-Step "Verifying imports"
    & $venvPython -c 'import fastapi, uvicorn, anthropic, keyring, pydantic; print("imports ok")'
    if ($LASTEXITCODE -ne 0) { throw "Sidecar imports failed inside the venv." }

    Write-Host ""
    Write-Ok "Setup complete."
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  * 2-run-dev.bat           - start the app with hot-reload"
    Write-Host "  * 3-build-installer.bat   - build the Windows installer"
    Write-Host ""
} catch {
    Write-Host ""
    Write-Err2 $_.Exception.Message
    Write-Host ""
    Write-Host "Setup did not finish cleanly. Read the message above for what to do." -ForegroundColor Yellow
    exit 1
}
