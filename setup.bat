@echo off
REM setup.bat — thin wrapper that delegates to setup.ps1.
REM
REM Usage: double-click on a clean Windows machine. PowerShell is included
REM in every supported Windows version, so no prerequisite is needed.

setlocal
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%setup.ps1" %*
set "ERR=%ERRORLEVEL%"
echo.
echo Press any key to close this window.
pause >nul
exit /b %ERR%
