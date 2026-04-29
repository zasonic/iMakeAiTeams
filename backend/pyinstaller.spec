# pyinstaller.spec — onedir bundle for the iMakeAiTeams Python sidecar.
#
# Produces backend/dist/server/server.exe + a sibling _internal/ tree.
# 3-build-installer.bat copies the whole dist/server directory into
# branding/sidecar-bundle/ so electron-builder can ship it as extraResources.
#
# Hidden imports cover:
#   * uvicorn's optional loops (`asyncio` is the default, but pyi misses
#     dynamically imported submodules)
#   * route modules in backend/routes/ — included via collect_submodules so
#     `app.include_router(routes.<module>)` resolves at runtime
#   * keyring backends — picked at runtime per-OS, the static analyzer
#     doesn't see them otherwise
#
# All third-party deps must ship as wheels — never compile at install time.

# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules, collect_all

block_cipher = None

# uvicorn ships shared libs and templates that PyInstaller's static analyzer
# doesn't auto-detect.
uvicorn_datas, uvicorn_binaries, uvicorn_hidden = collect_all("uvicorn")
fastapi_datas, fastapi_binaries, fastapi_hidden = collect_all("fastapi")
keyring_datas, keyring_binaries, keyring_hidden = collect_all("keyring")
anthropic_datas, anthropic_binaries, anthropic_hidden = collect_all("anthropic")

# All FastAPI route modules + every backend service module + everything under
# backend/core/ — collect_submodules walks the package recursively.
hidden_routes = collect_submodules("routes")
hidden_services = collect_submodules("services")
hidden_core = collect_submodules("core")

hiddenimports = (
    uvicorn_hidden
    + fastapi_hidden
    + keyring_hidden
    + anthropic_hidden
    + hidden_routes
    + hidden_services
    + hidden_core
    + [
        # uvicorn's default loop on Windows
        "uvicorn.loops.asyncio",
        "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan.on",
        # keyring native backends
        "keyring.backends.Windows",
        "keyring.backends.macOS",
        "keyring.backends.SecretService",
        "keyring.backends.fail",
        # legacy event-bus + smoke harness (referenced via dynamic import)
        "smoke_harness",
        "sse_events",
        "db",
        "models",
    ]
)

datas = uvicorn_datas + fastapi_datas + keyring_datas + anthropic_datas

binaries = uvicorn_binaries + fastapi_binaries + keyring_binaries + anthropic_binaries

a = Analysis(
    ["server.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Heavy deps belonging to the legacy 'full' variant. The lite installer
        # explicitly excludes them; service_guard.@requires reports them as
        # unavailable at runtime.
        # Note: PIL is intentionally NOT excluded — anthropic's SDK lazy-imports
        # it when the user attaches an image to a message.
        "torch",
        "sentence_transformers",
        "chromadb",
        "transformers",
        "tensorflow",
        "tkinter",
        "matplotlib",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="server",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="server",
)
