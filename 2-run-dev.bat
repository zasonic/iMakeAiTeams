@echo off
REM 2-run-dev.bat — start the app with hot-reload.
REM
REM Verifies 1-install.bat has been run (looks for backend\.venv and node_modules);
REM if either is missing, prints the recovery step instead of failing inside
REM npm/electron-vite.

setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if not exist "backend\.venv\Scripts\python.exe" (
    echo [error] backend\.venv is missing. Run 1-install.bat first.
    pause
    exit /b 1
)
if not exist "node_modules\electron-vite\package.json" (
    echo [error] node_modules is missing or incomplete. Run 1-install.bat first.
    pause
    exit /b 1
)

call npm run dev
exit /b %ERRORLEVEL%
