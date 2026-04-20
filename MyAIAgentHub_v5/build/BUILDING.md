# Building MyAI Agent Hub

The Windows shipping pipeline is **PyInstaller → Inno Setup → signtool**. There is no
runtime Python bootstrap anymore: the installer is the launcher.

---

## Windows

### Prerequisites

- Python 3.11 on PATH
- Inno Setup 6 with `iscc.exe` on PATH — <https://jrsoftware.org/isinfo.php>
- Microsoft Edge WebView2 Runtime **x64 offline installer** dropped at
  `build/webview2/MicrosoftEdgeWebView2RuntimeInstallerX64.exe`
  (download from <https://developer.microsoft.com/en-us/microsoft-edge/webview2/>).
  Gitignored — each build host supplies its own copy.

### Build

```bat
build\build_windows.bat full     REM ~1.7 GB installer: Tier 1 + Tier 2 + bundled model
build\build_windows.bat lite     REM ~60 MB installer:  Tier 1 only (chat, agents, teams, router)
```

What the script does:

1. `pip install` Tier 1 deps, plus Tier 2 for `full`.
2. `full` only: `python build/fetch_model.py` downloads `all-MiniLM-L6-v2` into
   `build/models/all-MiniLM-L6-v2/` for bundling. Fails the build if the
   download does not succeed — we do not ship without the model.
3. Verifies `build/webview2/MicrosoftEdgeWebView2RuntimeInstallerX64.exe` exists.
4. Runs `pyinstaller build/MyAIAgentHub.spec --noconfirm --clean`. The spec
   uses `collect_all()` for sentence_transformers, chromadb, anthropic,
   tokenizers, transformers, webview, and clr_loader — the full
   `(datas, binaries, hiddenimports)` triple, not just hidden imports.
5. `build/sign.ps1 <exe>` — no-op unless `MYAI_SIGN` is set.
6. `iscc build/installer.iss` (or `installer-lite.iss`) packages everything
   into `dist\MyAIAgentHub-Setup-Full.exe`. The Inno script conditionally
   invokes the bundled WebView2 installer from `[Run]` via `Check: NeedsWebView2`.
7. `build/sign.ps1 <installer>` — second pass of the two-pass sign.

Output:
- Full: `dist\MyAIAgentHub-Setup-Full.exe`
- Lite: `dist\MyAIAgentHub-Setup-Lite.exe`

### Launch log

Every launch writes one line to `%LOCALAPPDATA%\MyAIAgentHub\launch.log` from a
PyInstaller runtime hook (`build/runtime_hook_launch_log.py`) — before
`app/main.py` executes. Uncaught exceptions are also appended there. This is
the first place to check when diagnosing clean-VM failures.

### Microsoft Trusted Signing (optional)

Signing is gated on `MYAI_SIGN`. When unset, `build/sign.ps1` is a no-op — the
build produces unsigned artifacts. When set, both the inner EXE and the
installer are signed with SHA-256 and RFC 3161 timestamped via
`timestamp.acs.microsoft.com`. Required env:

```bat
set MYAI_SIGN=1
set TRUSTED_SIGNING_DLIB=C:\path\to\dir-containing-TrustedSigning.dll
set TRUSTED_SIGNING_METADATA=C:\path\to\signing-metadata.json
```

Verify after build:
```bat
signtool verify /pa /v dist\MyAIAgentHub-Setup-Full.exe
```

### Clean-VM verification (mandatory)

Four prior builds passed on the developer machine and failed in the wild.
The only real test is a clean VM. Before tagging a release:

1. **Pristine Windows 11 23H2 VM.** Hyper-V, VirtualBox, or Parallels.
   Fresh ISO install, no updates applied, no extra software. Snapshot → `BaseClean`.
2. Revert to `BaseClean`. Copy **only** `MyAIAgentHub-Setup-Full.exe` in —
   nothing else (no Python, no VC redist, no PATH edits).
3. **Disconnect the VM's network adapter.** The test from here is offline.
4. Double-click the installer. Complete with defaults.
5. Launch from Start Menu. Window must paint within ~10 s.
6. Confirm `%LOCALAPPDATA%\MyAIAgentHub\launch.log` exists with a fresh line.
7. Enter a real Anthropic key; send a chat message (reconnect briefly for this
   step, then disconnect again).
8. Upload a small PDF, ask a RAG question — **offline**. The bundled model
   must handle semantic search with no network.
9. Close, reopen — settings and chat history persist.
10. Uninstall. Confirm `C:\Program Files\MyAI Agent Hub\` is gone and
    `%LOCALAPPDATA%\MyAIAgentHub\` is **not** (user data preserved).

**Rule:** if any step fails, fix the build — not the VM.

---

## Mac

```bash
# Requires: pip install pyinstaller
bash build/build_mac.sh
```

Output: `dist/MyAI Agent Hub.app` (and optionally a `.dmg`).

Code-sign with an Apple Developer ID before distribution:
```bash
codesign --deep --force --verify --verbose \
  --sign "Developer ID Application: Your Name (XXXXXXXXXX)" \
  "dist/MyAI Agent Hub.app"
```

---

## App icons

| File | Format | Used by |
|------|--------|---------|
| `icons/AppIcon.icns` | macOS icon bundle | Mac builds |
| `icons/AppIcon.ico` | Windows icon | Windows builds |
| `icons/AppIcon.png` | 1024×1024 PNG | Source / Linux |

The spec fails early if the target-platform icon is missing.

---

## Notes

- **User data location**: `%LOCALAPPDATA%\MyAIAgentHub\` on Windows,
  `~/Library/Application Support/MyAIAgentHub/` on macOS. v5 installs stored
  this under `iMakeAiTeams/`; `paths._migrate_v5_user_dir()` moves it across
  the first time the new build runs.
- **Embedding model**: `all-MiniLM-L6-v2` is bundled inside the installer
  (full variant only, ~90 MB). It loads from
  `install_root()/_internal/models/all-MiniLM-L6-v2/` — no network on first run.
- **Lite variant** omits the model and all Tier 2 deps. `service_status()`
  reports semantic search and RAG as unavailable; the UI degrades gracefully.
