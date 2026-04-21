@echo off
REM build_windows.bat — Build MyAI Agent Hub for Windows.
REM
REM Usage:
REM   build\build_windows.bat            (default: full build)
REM   build\build_windows.bat lite       (Tier 1 only, ~60 MB installer)
REM   build\build_windows.bat full       (Tier 1 + Tier 2 + model, ~1.7 GB)
REM
REM Output:
REM   Full: dist\MyAIAgentHub\MyAIAgentHub.exe
REM         dist\MyAIAgentHub-Setup-Full.exe
REM   Lite: dist\MyAIAgentHub-lite\MyAIAgentHub-lite.exe
REM         dist\MyAIAgentHub-Setup-Lite.exe
REM
REM Requirements:
REM   pip install pyinstaller
REM   Inno Setup 6: https://jrsoftware.org/isinfo.php
REM   build\webview2\MicrosoftEdgeWebView2RuntimeInstallerX64.exe
REM     (download evergreen x64 offline installer from Microsoft)
REM
REM Optional signing (Microsoft Trusted Signing):
REM   set MYAI_SIGN=1
REM   set TRUSTED_SIGNING_DLIB=C:\path\to\dlib-dir
REM   set TRUSTED_SIGNING_METADATA=C:\path\to\signing-metadata.json

setlocal
cd /d "%~dp0\.."

REM Pick the variant — default to full.
set "VARIANT=%~1"
if "%VARIANT%"=="" set "VARIANT=full"
if /I not "%VARIANT%"=="lite" if /I not "%VARIANT%"=="full" (
    echo ERROR: variant must be 'lite' or 'full', got '%VARIANT%'
    exit /b 1
)
set MYAI_VARIANT=%VARIANT%

echo =^> Building %VARIANT% variant

echo =^> Installing/updating Tier 1 dependencies...
pip install -r app\requirements.txt --quiet || exit /b 1
pip install pyinstaller --quiet || exit /b 1

if /I "%VARIANT%"=="full" (
    echo =^> Installing Tier 2 extensions for full build...
    pip install -r app\requirements-extensions.txt --quiet || exit /b 1

    echo =^> Pre-downloading all-MiniLM-L6-v2 for bundling...
    python build\fetch_model.py
    if errorlevel 1 (
        echo ERROR: fetch_model.py failed. Cannot ship a full build without the model.
        exit /b 1
    )
)

REM Verify WebView2 offline installer is present before PyInstaller so we fail
REM early instead of after a 10-minute build.
if not exist "build\webview2\MicrosoftEdgeWebView2RuntimeInstallerX64.exe" (
    echo ERROR: build\webview2\MicrosoftEdgeWebView2RuntimeInstallerX64.exe is missing.
    echo        Download the x64 Evergreen Standalone Installer from
    echo        https://developer.microsoft.com/en-us/microsoft-edge/webview2/
    echo        and drop it in build\webview2\.
    exit /b 1
)

echo =^> Running PyInstaller (MYAI_VARIANT=%VARIANT%)...
pyinstaller build\MyAIAgentHub.spec --noconfirm --clean || exit /b 1

if /I "%VARIANT%"=="full" (
    set "EXE_PATH=dist\MyAIAgentHub\MyAIAgentHub.exe"
    set "ISS=build\installer.iss"
    set "SETUP=dist\MyAIAgentHub-Setup-Full.exe"
) else (
    set "EXE_PATH=dist\MyAIAgentHub-lite\MyAIAgentHub-lite.exe"
    set "ISS=build\installer-lite.iss"
    set "SETUP=dist\MyAIAgentHub-Setup-Lite.exe"
)

echo =^> Build output: %EXE_PATH%

echo =^> Running packaged smoke test against %EXE_PATH%...
set MYAI_PACKAGED_BINARY=%EXE_PATH%
python -m pytest tests\test_smoke_end_to_end.py::test_smoke_packaged -v || exit /b 1

REM Pass 1 of two-pass signing: sign the inner EXE before Inno Setup packages it.
REM sign.ps1 is a no-op unless MYAI_SIGN is set.
where pwsh >nul 2>&1
if %errorlevel% equ 0 (
    pwsh -NoProfile -ExecutionPolicy Bypass -File build\sign.ps1 "%EXE_PATH%" || exit /b 1
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File build\sign.ps1 "%EXE_PATH%" || exit /b 1
)

REM Build the installer with Inno Setup.
where iscc >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: iscc.exe not on PATH. Install Inno Setup 6:
    echo        https://jrsoftware.org/isinfo.php
    exit /b 1
)

echo =^> Building installer with Inno Setup...
iscc "%ISS%" || exit /b 1

REM Pass 2 of two-pass signing: sign the installer artifact.
where pwsh >nul 2>&1
if %errorlevel% equ 0 (
    pwsh -NoProfile -ExecutionPolicy Bypass -File build\sign.ps1 "%SETUP%" || exit /b 1
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File build\sign.ps1 "%SETUP%" || exit /b 1
)

echo.
echo Done. Installer: %SETUP%
endlocal
