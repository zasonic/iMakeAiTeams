# Quickstart

The fastest path from "I just unzipped this" to a running app.

## Windows (recommended path)

1. **Run `1-install.bat`** (double-click).
   It installs Node.js LTS and Python 3.12 if missing, creates `backend/.venv`,
   and runs `npm install`. First run takes 5–15 minutes depending on bandwidth.
   On failure it prints the exact recovery step in plain English.

2. **Run `2-run-dev.bat`** (double-click).
   This starts electron-vite in dev mode: the Electron window opens, the React
   UI hot-reloads, and the Python sidecar boots on a random localhost port.

3. **Open Settings inside the app** and paste your Anthropic API key.
   The key is stored in the Windows Credential Manager, not on disk.

4. **(Optional)** Install [Ollama](https://ollama.ai) and pull a model:
   ```
   ollama pull llama3:8b
   ```
   Then in Settings set your default local model. The router will send simple
   tasks there for free, and escalate to Claude when needed.

## Building a distributable installer

1. Make sure `1-install.bat` has been run successfully.
2. Run `3-build-installer.bat`.
3. Find the installer at `dist/iMakeAiTeams-Setup-<version>.exe`.
4. Test it on a clean VM (no Python, no Node) — the installer bundles
   everything end users need.

## macOS / Linux developers

See [`CONTRIBUTING.md`](CONTRIBUTING.md) — there's no `.bat` flow yet; you'll
run `npm install`, create the venv yourself, and use `npm run dev`.

## Where things end up

| What | Path (Windows) |
|---|---|
| Settings file | `%APPDATA%\MyAIAgentHub\settings.json` |
| Conversation database | `%APPDATA%\MyAIAgentHub\myai.db` |
| Main process log | `%APPDATA%\MyAIAgentHub\main.log` |
| Sidecar log | `%APPDATA%\MyAIAgentHub\sidecar.log` |
| API key | Windows Credential Manager (service: `iMakeAiTeams`) |

On macOS swap `%APPDATA%` for `~/Library/Application Support/`.

## Troubleshooting

**"backend\.venv is missing"** — `1-install.bat` didn't finish. Re-run it; the
script is safe to run multiple times.

**Sidecar fails to start** — open `%APPDATA%\MyAIAgentHub\sidecar.log` and
look at the last 50 lines. Common causes: port 5173/5174 in use, antivirus
blocking the PyInstaller-bundled exe, or missing Visual C++ redistributable.

**`npm run dev` exits immediately** — `node_modules/` is incomplete. Delete it
and re-run `1-install.bat`.

**App opens to a blank window** — likely a renderer build error. Check the
terminal where `2-run-dev.bat` is running for stack traces.

**Anthropic API errors** — your key is set in the app's Settings panel, not
in any `.env` file. Make sure it's pasted there and that your account has
quota remaining.
