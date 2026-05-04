@echo off
REM build-sidecar.bat — standalone PyInstaller invocation.
REM
REM Used by `npm run build:sidecar` and CI when the parent project doesn't want
REM to run the full electron-builder pipeline (3-build-installer.bat).
REM Outputs backend\dist\server\.

setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%..\backend"

if not exist ".venv\Scripts\python.exe" (
    echo [error] backend\.venv missing. Run 1-install.bat from the project root.
    exit /b 1
)

call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [error] could not activate venv
    exit /b 1
)

python -m PyInstaller pyinstaller.spec --noconfirm --clean
set "ERR=%ERRORLEVEL%"

call .venv\Scripts\deactivate.bat 2>nul
exit /b %ERR%
