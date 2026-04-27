@echo off
REM build_sidecar.bat — standalone PyInstaller invocation.
REM
REM Used by ship.bat (and CI) when the parent project doesn't want to run the
REM full electron-builder pipeline. Outputs backend\dist\server\.

setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if not exist ".venv\Scripts\python.exe" (
    echo [error] backend\.venv missing. Run setup.bat from the project root.
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
