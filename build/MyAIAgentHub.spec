# MyAIAgentHub.spec — PyInstaller build spec
# Usage:
#   pip install pyinstaller
#
#   # Full build (Tier 1 + Tier 2 ML deps + bundled all-MiniLM-L6-v2):
#   python build/fetch_model.py          # one-time, populates build/models/
#   MYAI_VARIANT=full pyinstaller build/MyAIAgentHub.spec --noconfirm --clean
#
#   # Lite build (Tier 1 only, ~60 MB — RAG/semantic search report unavailable):
#   MYAI_VARIANT=lite pyinstaller build/MyAIAgentHub.spec --noconfirm --clean
#
# Output: dist/MyAIAgentHub/MyAIAgentHub.exe (Windows) or .app (macOS).

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).parent       # repo root
APP  = ROOT / "app"

VARIANT = os.environ.get("MYAI_VARIANT", "full").lower()
if VARIANT not in ("lite", "full"):
    raise SystemExit(f"MYAI_VARIANT must be 'lite' or 'full', got {VARIANT!r}")

block_cipher = None

# ── Collection helper ────────────────────────────────────────────────────────
# collect_all() returns (datas, binaries, hiddenimports) — the full triple
# PyInstaller needs for packages that load data files (JSON configs, SQL DDL,
# pydantic type schemas) at runtime. Past builds failed because hiddenimports
# alone doesn't ship those non-.py files.

datas = []
binaries = []
hiddenimports = []

def _collect(pkg):
    d, b, h = collect_all(pkg)
    datas.extend(d)
    binaries.extend(b)
    hiddenimports.extend(h)

# Tier 1 — always required
for _pkg in ("anthropic", "webview", "clr_loader", "platformdirs",
             "pydantic", "tenacity", "psutil", "keyring", "numpy"):
    _collect(_pkg)

# Standard-lib-ish extras PyInstaller static analysis sometimes misses
hiddenimports.extend([
    "webview.platforms.cocoa",
    "webview.platforms.winforms",
    "webview.platforms.gtk",
    "numpy.core._multiarray_umath",
    "sqlite3", "json", "threading", "logging",
])

excludes = ["tkinter", "matplotlib", "PIL"]

# Tier 2 — bundled only in the full variant
if VARIANT == "full":
    for _pkg in ("sentence_transformers", "chromadb", "tokenizers",
                 "transformers", "rank_bm25", "torch"):
        _collect(_pkg)
else:
    # Lite: exclude heavy ML deps so PyInstaller doesn't transitively pull
    # them in via any surviving import reference.
    excludes.extend([
        "sentence_transformers", "chromadb", "tokenizers",
        "transformers", "rank_bm25", "torch",
    ])

# ── Bundled assets ───────────────────────────────────────────────────────────
# NEVER bundle settings.json or myai.db. Those are user data and live in
# paths.user_dir(); bundling them would ship read-only files that shadow the
# user copy inside the install dir after PyInstaller extracts.
datas.extend([
    (str(APP / "frontend"), "app/frontend"),
    (str(APP / "tools"),    "app/tools"),
])

# Bundle the embedding model so the app works offline on first launch.
if VARIANT == "full":
    MODEL_SRC = ROOT / "build" / "models" / "all-MiniLM-L6-v2"
    if not (MODEL_SRC / "config.json").exists():
        raise SystemExit(
            f"Model not found at {MODEL_SRC}. Run `python build/fetch_model.py` "
            f"before pyinstaller."
        )
    datas.append((str(MODEL_SRC), "models/all-MiniLM-L6-v2"))

# ── Icon ─────────────────────────────────────────────────────────────────────
_icon_path = (ROOT / "icons" / "AppIcon.ico") if sys.platform == "win32" \
             else (ROOT / "icons" / "AppIcon.icns")
if not _icon_path.exists():
    raise SystemExit(
        f"Icon not found at {_icon_path}. PyInstaller silently accepts bad "
        f"icon paths on non-Windows hosts; failing early instead."
    )

# ── Runtime hooks ────────────────────────────────────────────────────────────
# runtime_hook_launch_log.py runs before app/main.py executes so we always
# get a launch.log entry even on catastrophic pre-import failure.
_runtime_hooks = [str(ROOT / "build" / "runtime_hook_launch_log.py")]

a = Analysis(
    [str(APP / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=_runtime_hooks,
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
    icon=str(_icon_path),
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
        icon=str(_icon_path),
        bundle_identifier="com.imakeaiteams" + ("-lite" if VARIANT == "lite" else ""),
        info_plist={
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
            "CFBundleShortVersionString": "5.0.3",
            "LSMinimumSystemVersion": "13.0",
        },
    )
