@echo off
REM build_windows.bat — Build MyAI Agent Hub for Windows
REM
REM Produces: dist\MyAIAgentHub\MyAIAgentHub.exe
REM           (and optionally a setup installer via Inno Setup)
REM
REM Requirements:
REM   pip install pyinstaller
REM   Inno Setup 6 (optional, for .exe installer): https://jrsoftware.org/isinfo.php

setlocal
cd /d "%~dp0\.."

echo =^> Installing/updating dependencies...
pip install -r app\requirements.txt --quiet
pip install pyinstaller --quiet

echo =^> Running PyInstaller...
pyinstaller build\MyAIAgentHub.spec --noconfirm --clean

echo =^> Build output: dist\MyAIAgentHub\MyAIAgentHub.exe

REM Optional: build a proper installer with Inno Setup
where iscc >nul 2>&1
if %errorlevel% equ 0 (
  echo =^> Building installer with Inno Setup...
  iscc build\installer.iss
  echo =^> Installer: dist\MyAIAgentHub-Setup.exe
) else (
  echo.
  echo Tip: install Inno Setup to produce a one-click .exe installer:
  echo   https://jrsoftware.org/isinfo.php
  echo   Then re-run this script.
)

echo.
echo Done. Run: dist\MyAIAgentHub\MyAIAgentHub.exe
endlocal
