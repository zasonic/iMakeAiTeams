@echo off
REM Start.bat — single user-facing entry point for iMakeAiTeams dev.
REM
REM First run (no node_modules) installs prerequisites via dev\install.ps1
REM and then launches the app. Subsequent runs go straight to npm run dev.

setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if not exist "node_modules\electron-vite\package.json" (
    echo ==^> First run detected. Installing prerequisites...
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%dev\install.ps1" %*
    if errorlevel 1 (
        echo.
        echo Setup did not finish. Read the message above for what to do.
        pause
        exit /b 1
    )
)

if not exist "backend\.venv\Scripts\python.exe" (
    echo [error] backend\.venv is missing. Re-run Start.bat to retry install.
    pause
    exit /b 1
)
if not exist "node_modules\electron-vite\package.json" (
    echo [error] node_modules is missing or incomplete. Re-run Start.bat.
    pause
    exit /b 1
)

call npm run dev
exit /b %ERRORLEVEL%
