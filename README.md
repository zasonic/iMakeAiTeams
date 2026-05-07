# iMakeAiTeams

Build AI teams that work together on your desktop.

## What this does

Chat with Claude and local AI models in a single interface. Create
specialist agents with custom instructions and personalities. Index your
files and folders so agents can reference them in conversation. Everything
runs locally — your data stays on your machine.

## Download

Download the latest installer from
[Releases](https://github.com/zasonic/iMakeAiTeams/releases). Double-click
to install. No setup required.

## What you need

- An [Anthropic API key](https://console.anthropic.com/settings/keys) — required.
- [Ollama](https://ollama.com/download) — recommended. Lets simple
  messages run locally for free instead of every message going to Claude.
- Or [LM Studio](https://lmstudio.ai/) — auto-detected if you already use it.

## What you'll see

- **Chat** — Talk to your AI team; messages route to the best model automatically.
- **Agents** — Create specialists with custom instructions and personalities.
- **Documents** — Add files and folders for your team to reference.
- **Memory** — Your team remembers facts across conversations.
- **Settings** — API keys, models, and routing controls.

### How routing works

Each message is classified before it is sent. Simple requests
(greetings, formatting, short answers) go to your local model when one
is available; complex requests (analysis, planning, code review) go to
Claude. If no local model is running, every message routes to Claude
and a "Local model offline" indicator appears in the status bar so you
know costs are higher.

### How search works

Document indexing and semantic memory run entirely on your machine
using fastembed (ONNX) and sqlite-vec. No text leaves your computer
for embedding. The index lives next to the rest of the app data
(see "Where your data lives").

Toggle Studio Mode at the bottom of the sidebar for advanced features:
prompt engineering, MCP tool servers, security scanning, and diagnostics.

## Where your data lives

- Settings and database in `%APPDATA%/iMakeAiTeams/` on Windows,
  `~/Library/Application Support/iMakeAiTeams/` on macOS,
  `~/.config/iMakeAiTeams/` on Linux.
- Document and memory vectors live inside the same SQLite database
  (sqlite-vec extension) — no separate vector store directory.
- Logs in the same folder.
- API key stored in the OS keyring (Windows Credential Manager / macOS
  Keychain / Linux SecretService), not on disk.

## For developers

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, architecture, and build instructions.

On Windows, double-click `Start.bat` to install prerequisites (first run) and
launch the app with hot-reload. Run `dev\build-installer.bat` to produce the
NSIS installer.

## License

MIT — see [LICENSE](LICENSE).
