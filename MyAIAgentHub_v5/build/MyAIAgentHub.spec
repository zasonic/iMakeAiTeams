# MyAIAgentHub.spec — PyInstaller build spec
# Usage:
#   pip install pyinstaller
#   pyinstaller build/MyAIAgentHub.spec
#
# Output: dist/MyAIAgentHub.app  (Mac) or dist/MyAIAgentHub.exe (Windows)

import sys
from pathlib import Path

ROOT = Path(SPECPATH).parent       # repo root
APP  = ROOT / "app"

block_cipher = None

a = Analysis(
    [str(APP / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        # Bundle the entire frontend
        (str(APP / "frontend"), "app/frontend"),
        # Bundle default settings
        (str(APP / "settings.json"), "app"),
        # Keep tools directory
        (str(APP / "tools"), "app/tools"),
    ],
    hiddenimports=[
        # Anthropic SDK
        "anthropic", "anthropic._models", "anthropic.types",
        # PyWebView per-platform backends
        "webview.platforms.cocoa",       # macOS
        "webview.platforms.winforms",    # Windows
        "webview.platforms.gtk",         # Linux
        # ML/vector
        "sentence_transformers",
        "sentence_transformers.models",
        "chromadb",
        "chromadb.db.impl.sqlite",
        # Numerics
        "numpy", "numpy.core._multiarray_umath",
        # Standard libs sometimes missed
        "sqlite3", "json", "threading", "logging",
        "psutil", "tenacity", "pydantic",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PIL"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="MyAIAgentHub",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,                   # no terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=str(ROOT / "icons" / "AppIcon.ico") if sys.platform == "win32"
         else str(ROOT / "icons" / "AppIcon.icns"),
)

coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=True, upx_exclude=[],
    name="MyAIAgentHub",
)

# macOS .app bundle
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="MyAI Agent Hub.app",
        icon=str(ROOT / "icons" / "AppIcon.icns"),
        bundle_identifier="com.myaiagenthub",
        info_plist={
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
            "CFBundleShortVersionString": "1.0.0",
            "LSMinimumSystemVersion": "13.0",
        },
    )
