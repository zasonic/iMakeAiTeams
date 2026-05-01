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

- An [Anthropic API key](https://console.anthropic.com/settings/keys).
- Optionally, [Ollama](https://ollama.ai) for free local inference.

## What you'll see

- **Chat** — Talk to your AI team; messages route to the best model automatically.
- **Agents** — Create specialists with custom instructions and personalities.
- **Documents** — Add files and folders for your team to reference.
- **Memory** — Your team remembers facts across conversations.
- **Settings** — API keys, models, and routing controls.

Toggle Studio Mode at the bottom of the sidebar for advanced features:
prompt engineering, MCP tool servers, security scanning, and diagnostics.

## Where your data lives

- Settings and database in `%APPDATA%/iMakeAiTeams/` on Windows,
  `~/Library/Application Support/iMakeAiTeams/` on macOS.
- Logs in the same folder.
- API key stored in the OS keyring (Windows Credential Manager / macOS
  Keychain), not on disk.

## For developers

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, architecture, and build instructions.

See [QUICKSTART.md](QUICKSTART.md) for the developer quickstart using batch files.

## License

MIT — see [LICENSE](LICENSE).
