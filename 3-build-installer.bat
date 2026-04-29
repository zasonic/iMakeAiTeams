@echo off
REM 3-build-installer.bat — full installer build pipeline.
REM
REM 1. Activate backend venv
REM 2. PyInstaller --onedir into backend\dist\server\
REM 3. Mirror that to branding\sidecar-bundle\ for electron-builder extraResources
REM 4. electron-vite production build
REM 5. electron-builder --win  -> NSIS installer in dist\

setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if not exist "backend\.venv\Scripts\python.exe" (
    echo [error] backend\.venv is missing. Run 1-install.bat first.
    pause
    exit /b 1
)
if not exist "node_modules\electron-builder\package.json" (
    echo [error] node_modules is missing. Run 1-install.bat first.
    pause
    exit /b 1
)

echo ==^> [1/5] Building Python sidecar with PyInstaller (onedir)
pushd backend
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [error] could not activate venv
    popd
    pause
    exit /b 1
)
python -m PyInstaller pyinstaller.spec --noconfirm --clean
set "PYI_ERR=%ERRORLEVEL%"
call .venv\Scripts\deactivate.bat 2>nul
popd
if not "%PYI_ERR%"=="0" (
    echo [error] PyInstaller failed with exit code %PYI_ERR%
    pause
    exit /b %PYI_ERR%
)

echo ==^> [2/5] Mirroring sidecar to branding\sidecar-bundle\
if exist "branding\sidecar-bundle" rmdir /s /q "branding\sidecar-bundle"
mkdir "branding\sidecar-bundle"
xcopy "backend\dist\server" "branding\sidecar-bundle" /e /i /q /y >nul
if errorlevel 1 (
    echo [error] xcopy failed
    pause
    exit /b 1
)

echo ==^> [3/5] electron-vite production build
call npm run build
if errorlevel 1 (
    echo [error] electron-vite build failed
    pause
    exit /b 1
)

echo ==^> [4/5] electron-builder NSIS installer
call npx --no-install electron-builder --win
if errorlevel 1 (
    echo [error] electron-builder failed
    pause
    exit /b 1
)

echo ==^> [5/5] Done.
echo.
for %%f in (dist\*Setup*.exe) do (
    echo Installer: %%~ff
)
echo.
echo Test the installer on a clean Windows VM (no Python, no Node).
echo The user double-clicks the .exe; no terminal required.
echo.
pause
exit /b 0
