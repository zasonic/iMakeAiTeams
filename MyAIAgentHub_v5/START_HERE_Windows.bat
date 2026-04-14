@echo off
setlocal enabledelayedexpansion

:: ═══════════════════════════════════════════════════════════════════
::  START HERE — MyAI Agent Hub (Windows)
::  Double-click this file. Everything else is automatic.
:: ═══════════════════════════════════════════════════════════════════

set "APP=%~dp0app"

:: ── 1. Check for portable Python (from a previous bootstrap) ─────
if exist "%APP%\.python\python.exe" (
    set "PYEXE=%APP%\.python\python.exe"
    goto :launch
)

:: ── 2. Check for system Python ───────────────────────────────────
where py >nul 2>&1 && (
    for /f "tokens=*" %%i in ('py -c "import sys; print(sys.executable)"') do set "PYEXE=%%i"
    goto :launch
)
where python >nul 2>&1 && (
    python -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" 2>nul && (
        set "PYEXE=python"
        goto :launch
    )
)
where python3 >nul 2>&1 && (
    set "PYEXE=python3"
    goto :launch
)

:: ── 3. No Python found — bootstrap a portable copy ──────────────
echo.
echo  ┌─────────────────────────────────────────────────────────┐
echo  │  Python not found. Downloading a portable copy...       │
echo  │  (One-time setup, ~25 MB download)                      │
echo  └─────────────────────────────────────────────────────────┘
echo.

:: bootstrap_python.ps1 is a local file, so RemoteSigned is sufficient.
:: -ExecutionPolicy Bypass overrides all user security settings and is
:: a pattern actively flagged by Windows Defender and antivirus software.
powershell -ExecutionPolicy RemoteSigned -NoProfile -File "%APP%\bootstrap_python.ps1" "%APP%"

if errorlevel 1 (
    echo.
    echo  Setup failed. Please install Python 3.10+ manually:
    echo  https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

set "PYEXE=%APP%\.python\python.exe"

:: ── 4. Launch ────────────────────────────────────────────────────
:launch

:: Prefer pythonw (no console window) but fall back to python
if "%PYEXE%"=="python" (
    where pythonw >nul 2>&1 && (
        start "" pythonw "%APP%\setup_launcher.pyw"
        exit /b 0
    )
)
if "%PYEXE%"=="python3" (
    start "" python3 "%APP%\setup_launcher.pyw"
    exit /b 0
)

:: For portable or absolute paths, check for pythonw next to python
set "PYWEXE=%PYEXE:python.exe=pythonw.exe%"
if exist "%PYWEXE%" (
    start "" "%PYWEXE%" "%APP%\setup_launcher.pyw"
) else (
    start "" "%PYEXE%" "%APP%\setup_launcher.pyw"
)
exit /b 0
