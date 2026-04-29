# iMakeAiTeams

A local-first desktop app where Claude and local models work as a coordinated team.
Electron shell + React UI + Python (FastAPI) sidecar.

## What's in this folder

| Path | What it is |
|---|---|
| `desktop-ui/` | React renderer — the windows, panels, and chat UI |
| `desktop-shell/` | Electron main + preload — the desktop app host |
| `backend/` | Python FastAPI sidecar — chat orchestration, RAG, agents, MCP |
| `branding/` | App icon and bundled-sidecar staging directory |
| `build-scripts/` | Helpers invoked by `npm run` (sidecar build, packaging) |
| `archive/legacy-v5/` | Historical snapshot of the predecessor app. Not used at runtime. |
| `1-install.bat` / `.ps1` | First-time Windows setup (Node + Python + dependencies) |
| `2-run-dev.bat` | Start the app with hot-reload |
| `3-build-installer.bat` | Build the Windows NSIS installer |

## Prerequisites

- **Windows 10/11** (Mac/Linux dev possible — see `CONTRIBUTING.md`)
- **Node.js 20+** and **Python 3.12+** (the installer script can fetch both)
- **Anthropic API key** — entered in the app's Settings on first launch
- **Optional:** [Ollama](https://ollama.ai) for free local inference

## Quickstart (Windows)

```text
1. Double-click  1-install.bat   (one-time; installs Node, Python, npm + pip deps)
2. Double-click  2-run-dev.bat   (starts Electron + React + sidecar with hot-reload)
3. In the app    Settings -> paste your Anthropic API key
```

To produce a distributable installer: double-click `3-build-installer.bat`.
The result lands in `dist/iMakeAiTeams-Setup-<version>.exe`.

See [`QUICKSTART.md`](QUICKSTART.md) for a detailed walkthrough and troubleshooting,
or [`CONTRIBUTING.md`](CONTRIBUTING.md) for the dev-from-source flow on Mac/Linux.

## Architecture in one paragraph

The Electron shell (`desktop-shell/`) owns the window and spawns the Python sidecar
on a random localhost port with a per-launch bearer token. The React renderer
(`desktop-ui/`) talks to the sidecar over HTTP + Server-Sent Events; all network
traffic is 127.0.0.1. The sidecar (`backend/`) routes each message between Claude
and a local model, manages multi-agent teams and workflows, and persists state to
your OS user-data directory (`%APPDATA%/MyAIAgentHub` on Windows).

## Where things live at runtime

- **Settings + database:** `%APPDATA%/MyAIAgentHub/` (Windows) or
  `~/Library/Application Support/MyAIAgentHub/` (macOS)
- **Logs:** same folder, `main.log` and `sidecar.log`
- **API key:** stored in the OS keyring, not on disk
