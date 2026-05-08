@echo off
REM dev\run-bench.bat — local AgentDojo bench runner.
REM
REM Activates the project venv, installs the bench-only deps from
REM backend\requirements-bench.txt, runs all four suites, and regenerates
REM BENCHMARKS.md. The workflow that runs in CI lives at
REM .github\workflows\security-bench.yml — this script mirrors its steps
REM so contributors can reproduce the published numbers locally.
REM
REM Pre-reqs:
REM   * dev\install.ps1 has run successfully (creates backend\.venv).
REM   * ANTHROPIC_API_KEY is set in your shell before invoking this script.

setlocal enableextensions enabledelayedexpansion

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"

set "VENV_PY=%REPO_ROOT%\backend\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [error] backend\.venv\Scripts\python.exe not found.
    echo         Run dev\install.ps1 first to create the venv.
    popd
    exit /b 1
)

if "%ANTHROPIC_API_KEY%"=="" (
    echo [error] ANTHROPIC_API_KEY env var is not set.
    echo         Set it in your shell, e.g.:
    echo           set ANTHROPIC_API_KEY=sk-ant-...
    popd
    exit /b 1
)

echo ==^> Installing bench dependencies
"%VENV_PY%" -m pip install --timeout=1000 --retries=20 --no-cache-dir -r backend\requirements-bench.txt
if errorlevel 1 (
    echo [error] pip install -r backend\requirements-bench.txt failed.
    popd
    exit /b 1
)

if not exist "%REPO_ROOT%\benchmarks" mkdir "%REPO_ROOT%\benchmarks"

for %%S in (workspace slack banking travel) do (
    echo ==^> Running AgentDojo suite: %%S
    "%VENV_PY%" -m backend.tests.agentdojo.run_suites --suite %%S --output benchmarks\%%S.json
    if errorlevel 1 (
        echo [error] suite %%S failed.
        popd
        exit /b 1
    )
)

echo ==^> Regenerating BENCHMARKS.md
"%VENV_PY%" build-scripts\generate_benchmarks_md.py --benchmarks-dir benchmarks --thresholds benchmarks\thresholds.json --output BENCHMARKS.md
if errorlevel 1 (
    echo [error] BENCHMARKS.md generation failed.
    popd
    exit /b 1
)

echo [ok] Bench run complete. See BENCHMARKS.md and benchmarks\*.json.
popd
endlocal
