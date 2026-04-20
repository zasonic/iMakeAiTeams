# Building MyAI Agent Hub

Two build approaches are available: **Briefcase** (recommended for final distribution) and **PyInstaller** (faster iteration).

---

## Option A — Briefcase (recommended)

Briefcase produces proper platform-native installers: a `.dmg` on Mac and an `.msi` on Windows.

### Install Briefcase

```bash
pip install briefcase
```

### Mac

```bash
# First time: scaffold the platform-specific project
briefcase create macOS

# Build the .app bundle
briefcase build macOS

# Package into a distributable .dmg
briefcase package macOS
# Output: dist/MyAI Agent Hub-1.0.0.dmg
```

### Windows

```bash
briefcase create windows
briefcase build windows
briefcase package windows
# Output: dist/MyAI Agent Hub-1.0.0.msi
# Note: Requires WiX Toolset 3.x — https://wixtoolset.org/
```

### Run in dev mode (no build step)

```bash
briefcase dev
```

---

## Option B — PyInstaller (faster, no installer polish)

Produces a `.app` bundle (Mac) or an `.exe` + folder (Windows) without a system installer.

### Mac

```bash
# Requires: pip install pyinstaller
# Optional: brew install create-dmg   (for .dmg output)
bash build/build_mac.sh
```

Output: `dist/MyAI Agent Hub.app` and optionally `dist/MyAIAgentHub-1.0.0-mac.dmg`

### Windows

Two variants are supported — pick one based on whether document search and
semantic memory are needed:

```bash
build\build_windows.bat full    REM ~1.6 GB installer: Tier 1 + Tier 2 (RAG, semantic search, BM25)
build\build_windows.bat lite    REM ~60 MB installer:  Tier 1 only (chat, agents, teams, router)
```

Output:
- Full: `dist\MyAIAgentHub\MyAIAgentHub.exe` + `dist\MyAIAgentHub-Setup-Full.exe`
- Lite: `dist\MyAIAgentHub-lite\MyAIAgentHub-lite.exe` + `dist\MyAIAgentHub-Setup-Lite.exe`

The lite build uses the same source tree — Tier 2 imports are lazy (see
`core/api.py:95`, `services/semantic_search.py:58,93`, `channels/telegram_adapter.py:90`),
so the lite installer bundles no PyTorch or sentence-transformers. At runtime,
`service_status()` reports Tier 2 services as unavailable and the Settings →
Subsystem status panel makes this visible to the user.

Both installers write user data to `%LOCALAPPDATA%\iMakeAiTeams\` (resolved by
`core/paths.py`), never into `Program Files` — so a lite→full upgrade preserves
conversations, settings, and the OS-keyring-stored API key automatically.

For single-file setup installers, install [Inno Setup 6](https://jrsoftware.org/isinfo.php)
first — the script detects it automatically and runs `iscc build\installer.iss`
(full) or `iscc build\installer-lite.iss` (lite).

---

## App icon

Both build methods expect icon files at:

| File | Format | Used by |
|------|--------|---------|
| `icons/AppIcon.icns` | macOS icon bundle | Mac builds |
| `icons/AppIcon.ico` | Windows icon | Windows builds |
| `icons/AppIcon.png` | 1024×1024 PNG | Source / Linux |

Create the `.icns` from a 1024×1024 PNG on Mac:
```bash
mkdir -p icons/AppIcon.iconset
sips -z 1024 1024 icons/AppIcon.png --out icons/AppIcon.iconset/icon_512x512@2x.png
iconutil -c icns icons/AppIcon.iconset -o icons/AppIcon.icns
```

Convert to `.ico` on any platform:
```bash
pip install Pillow
python3 -c "
from PIL import Image
img = Image.open('icons/AppIcon.png')
img.save('icons/AppIcon.ico', sizes=[(16,16),(32,32),(48,48),(256,256)])
"
```

---

## Notes

- **Sentence-transformers model** (`all-MiniLM-L6-v2`, ~90MB) downloads automatically on first launch from the bundled app. This requires an internet connection on the first run but is cached afterward.
- **WebView2 on Windows**: the app uses Microsoft Edge WebView2, which ships with Windows 11 and is auto-installed on Windows 10. If a user is missing it, the PyWebView startup error message will tell them.
- **Code signing**: unsigned `.app` bundles will trigger Gatekeeper on Mac. For a proper distribution, sign with an Apple Developer certificate:
  ```bash
  codesign --deep --force --verify --verbose \
    --sign "Developer ID Application: Your Name (XXXXXXXXXX)" \
    "dist/MyAI Agent Hub.app"
  ```
