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
fastembed_datas, fastembed_binaries, fastembed_hidden = collect_all("fastembed")
onnxruntime_datas, onnxruntime_binaries, onnxruntime_hidden = collect_all("onnxruntime")
sqlite_vec_datas, sqlite_vec_binaries, sqlite_vec_hidden = collect_all("sqlite_vec")

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
    + fastembed_hidden
    + onnxruntime_hidden
    + sqlite_vec_hidden
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

# Phase 6: bundle backend/templates/ so reader_system.txt / actor_system.txt
# (and the existing Caddyfile/docker-compose Jinja templates) ship in the
# PyInstaller onedir output. Without this the static analyzer skips .txt/.j2
# files and prompt loading fails at runtime in the installed app.
datas = (
    uvicorn_datas + fastapi_datas + keyring_datas + anthropic_datas
    + fastembed_datas + onnxruntime_datas + sqlite_vec_datas
    + [("templates", "templates")]
)

binaries = uvicorn_binaries + fastapi_binaries + keyring_binaries + anthropic_binaries + fastembed_binaries + onnxruntime_binaries + sqlite_vec_binaries

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
        # NOT dependencies of the current stack. Excludes prevent
        # accidental transitive imports from fastembed or onnxruntime.
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
    console=False,   # suppress terminal window in production
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
