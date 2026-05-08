# iMakeAiTeams

Build AI teams that work together on your desktop. Chat with Claude and
local models, create specialist agents, index your files, and keep
everything on your machine.

## Install

Download the latest installer from
[Releases](https://github.com/zasonic/iMakeAiTeams/releases) and double-click.
No setup required.

You will need:
- An [Anthropic API key](https://console.anthropic.com/settings/keys).
- [Ollama](https://ollama.com/download) or [LM Studio](https://lmstudio.ai/)
  for local models (optional but recommended — keeps simple messages free).

## Run

Open the app, paste your API key in Settings, and start chatting. Messages
route automatically: simple turns go to a local model when one is available,
complex turns go to Claude.

For developers (Windows): double-click `Start.bat`. First run installs
Node, Python, and dependencies; subsequent runs go straight to dev mode.

```
Start.bat                    # install + dev (Windows)
npm run dev                  # cross-platform dev (after manual setup)
dev\build-installer.bat      # produce NSIS installer (Windows)
```

## Where things live

- App data: `%APPDATA%/iMakeAiTeams/` on Windows,
  `~/Library/Application Support/iMakeAiTeams/` on macOS,
  `~/.config/iMakeAiTeams/` on Linux. Settings, SQLite database, and logs
  all live there. The API key is in the OS keyring, not on disk.
- Source layout: `desktop-ui/` (React renderer), `desktop-shell/` (Electron
  main + preload), `backend/` (Python FastAPI sidecar), `branding/` (icon +
  staged sidecar bundle), `dev/` (developer scripts).

Deeper docs:
- [docs/USER-GUIDE.md](docs/USER-GUIDE.md) — features and how to use them.
- [docs/architecture.md](docs/architecture.md) — services, schemas, IPC.
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — dev setup and build flow.
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — common errors.
- [docs/FAQ.md](docs/FAQ.md) — common questions.
- [docs/legacy.md](docs/legacy.md) — pre-v6 code lives on the legacy/v5 branch.
- [BENCHMARKS.md](BENCHMARKS.md) — AgentDojo ASR vs the published baseline, refreshed every push to `main`.

MIT — see [LICENSE](LICENSE).
