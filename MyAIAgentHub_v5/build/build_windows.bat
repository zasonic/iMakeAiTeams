@echo off
REM build_windows.bat — Build MyAI Agent Hub for Windows.
REM
REM Usage:
REM   build\build_windows.bat            (default: full build)
REM   build\build_windows.bat lite       (Tier 1 only, ~60 MB installer)
REM   build\build_windows.bat full       (Tier 1 + Tier 2, ~1.6 GB installer)
REM
REM Output:
REM   Full: dist\MyAIAgentHub\MyAIAgentHub.exe
REM         dist\MyAIAgentHub-Setup-Full.exe
REM   Lite: dist\MyAIAgentHub-lite\MyAIAgentHub-lite.exe
REM         dist\MyAIAgentHub-Setup-Lite.exe
REM
REM Requirements:
REM   pip install pyinstaller
REM   Inno Setup 6 (optional, for .exe installer): https://jrsoftware.org/isinfo.php

setlocal
cd /d "%~dp0\.."

REM Pick the variant — default to full for backwards compatibility.
set "VARIANT=%~1"
if "%VARIANT%"=="" set "VARIANT=full"
if /I not "%VARIANT%"=="lite" if /I not "%VARIANT%"=="full" (
    echo ERROR: variant must be 'lite' or 'full', got '%VARIANT%'
    exit /b 1
)
set MYAI_VARIANT=%VARIANT%

echo =^> Building %VARIANT% variant

echo =^> Installing/updating Tier 1 dependencies...
pip install -r app\requirements.txt --quiet
pip install pyinstaller --quiet

if /I "%VARIANT%"=="full" (
    echo =^> Installing Tier 2 extensions for full build...
    pip install -r app\requirements-extensions.txt --quiet
)

echo =^> Running PyInstaller (MYAI_VARIANT=%VARIANT%)...
pyinstaller build\MyAIAgentHub.spec --noconfirm --clean

if /I "%VARIANT%"=="full" (
    echo =^> Build output: dist\MyAIAgentHub\MyAIAgentHub.exe
    set "ISS=build\installer.iss"
    set "SETUP=dist\MyAIAgentHub-Setup-Full.exe"
) else (
    echo =^> Build output: dist\MyAIAgentHub-lite\MyAIAgentHub-lite.exe
    set "ISS=build\installer-lite.iss"
    set "SETUP=dist\MyAIAgentHub-Setup-Lite.exe"
)

REM Optional: build a proper installer with Inno Setup
where iscc >nul 2>&1
if %errorlevel% equ 0 (
    echo =^> Building installer with Inno Setup...
    iscc "%ISS%"
    echo =^> Installer: %SETUP%
) else (
    echo.
    echo Tip: install Inno Setup to produce a one-click .exe installer:
    echo   https://jrsoftware.org/isinfo.php
    echo   Then re-run this script.
)

echo.
echo Done.
endlocal
