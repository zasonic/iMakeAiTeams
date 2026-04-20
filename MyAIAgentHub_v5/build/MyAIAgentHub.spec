# MyAIAgentHub.spec — PyInstaller build spec
# Usage:
#   pip install pyinstaller
#
#   # Full build (~1.6 GB — Tier 1 + Tier 2 ML deps):
#   MYAI_VARIANT=full pyinstaller build/MyAIAgentHub.spec
#
#   # Lite build (~60 MB — Tier 1 only, RAG + semantic search reported
#   # unavailable by service_status() at runtime; app still works):
#   MYAI_VARIANT=lite pyinstaller build/MyAIAgentHub.spec
#
# Output: dist/MyAIAgentHub/MyAIAgentHub.exe (Windows) or .app (macOS).

import os
import sys
from pathlib import Path

ROOT = Path(SPECPATH).parent       # repo root
APP  = ROOT / "app"

VARIANT = os.environ.get("MYAI_VARIANT", "full").lower()
if VARIANT not in ("lite", "full"):
    raise SystemExit(f"MYAI_VARIANT must be 'lite' or 'full', got {VARIANT!r}")

block_cipher = None

# ── Hidden imports ───────────────────────────────────────────────────────────
# Tier 1 — always required
_tier1_imports = [
    # Anthropic SDK — internal modules PyInstaller sometimes misses
    "anthropic", "anthropic._models", "anthropic.types",
    # PyWebView per-platform backends
    "webview.platforms.cocoa",
    "webview.platforms.winforms",
    "webview.platforms.gtk",
    # Numerics
    "numpy", "numpy.core._multiarray_umath",
    # Standard libs sometimes missed by PyInstaller's static analysis
    "sqlite3", "json", "threading", "logging",
    "psutil", "tenacity", "pydantic", "platformdirs",
]

# Tier 2 — bundled only in the full variant
_tier2_imports = [
    "sentence_transformers",
    "sentence_transformers.models",
    "chromadb",
    "chromadb.db.impl.sqlite",
    "rank_bm25",
    "torch",
]

hiddenimports = list(_tier1_imports)
excludes = ["tkinter", "matplotlib", "PIL"]
if VARIANT == "full":
    hiddenimports.extend(_tier2_imports)
else:
    # Lite: exclude heavy ML deps so PyInstaller doesn't transitively pull them
    # in via any surviving import reference.
    excludes.extend(_tier2_imports)

# ── Data files ───────────────────────────────────────────────────────────────
# NEVER bundle settings.json or myai.db here. Those are user data and must
# live in paths.user_dir(); bundling them would ship read-only files that
# shadow the user copy inside the install dir after PyInstaller extracts.
datas = [
    (str(APP / "frontend"), "app/frontend"),
    (str(APP / "tools"), "app/tools"),
]

a = Analysis(
    [str(APP / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

_exe_name = f"MyAIAgentHub-{VARIANT}" if VARIANT == "lite" else "MyAIAgentHub"

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name=_exe_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=str(ROOT / "icons" / "AppIcon.ico") if sys.platform == "win32"
         else str(ROOT / "icons" / "AppIcon.icns"),
)

coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=True, upx_exclude=[],
    name=_exe_name,
)

# macOS .app bundle
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"MyAI Agent Hub{' Lite' if VARIANT == 'lite' else ''}.app",
        icon=str(ROOT / "icons" / "AppIcon.icns"),
        bundle_identifier="com.imakeaiteams" + ("-lite" if VARIANT == "lite" else ""),
        info_plist={
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
            "CFBundleShortVersionString": "5.0.2",
            "LSMinimumSystemVersion": "13.0",
        },
    )
