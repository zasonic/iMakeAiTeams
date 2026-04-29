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
REM Invoke the venv python directly — `call activate.bat` swallows errors and
REM can leave PATH pointing at the system Python on some Windows configurations.
pushd backend
".venv\Scripts\python.exe" -m PyInstaller pyinstaller.spec --noconfirm --clean
set "PYI_ERR=%ERRORLEVEL%"
popd
if not "%PYI_ERR%"=="0" (
    echo [error] PyInstaller failed with exit code %PYI_ERR%
    pause
    exit /b %PYI_ERR%
)
if not exist "backend\dist\server\server.exe" (
    echo [error] PyInstaller exited 0 but backend\dist\server\server.exe is missing
    pause
    exit /b 1
)

echo ==^> [2/5] Mirroring sidecar to branding\sidecar-bundle\
if exist "branding\sidecar-bundle" (
    rmdir /s /q "branding\sidecar-bundle"
    if errorlevel 1 (
        echo [error] could not remove existing branding\sidecar-bundle (file in use?)
        pause
        exit /b 1
    )
)
mkdir "branding\sidecar-bundle"
xcopy "backend\dist\server" "branding\sidecar-bundle" /e /i /q /y >nul
if errorlevel 1 (
    echo [error] xcopy failed
    pause
    exit /b 1
)
if not exist "branding\sidecar-bundle\server.exe" (
    echo [error] mirror finished but branding\sidecar-bundle\server.exe is missing
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
set "FOUND_INSTALLER="
for %%f in (dist\*.exe) do (
    echo Installer: %%~ff
    set "FOUND_INSTALLER=1"
)
if not defined FOUND_INSTALLER (
    echo [error] electron-builder reported success but no .exe was produced in dist\
    pause
    exit /b 1
)
echo.
echo Test the installer on a clean Windows VM (no Python, no Node).
echo The user double-clicks the .exe; no terminal required.
echo.
pause
exit /b 0
