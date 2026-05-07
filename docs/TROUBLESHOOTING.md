# Troubleshooting

What to do when things break.

## "Anthropic API errors" / "Claude API key invalid"

Your key is set in the app's Settings panel, not in any `.env` file.
Open Settings → Anthropic API key, paste your key, click Verify & save.
The key is stored in the OS keyring (Windows Credential Manager / macOS
Keychain / Linux SecretService) — not on disk.

If verify fails:
- Check that the key starts with `sk-ant-`.
- Confirm your account has remaining quota at
  [console.anthropic.com](https://console.anthropic.com/settings/billing).
- Make sure your firewall isn't blocking outbound HTTPS.

## "LM Studio not found" / "Local model offline"

The status bar shows "Local model offline" when neither Ollama nor LM
Studio is reachable. Every message routes to Claude in that state, so
costs are higher.

To fix:
- Install [Ollama](https://ollama.com/download) and pull a model:
  `ollama pull llama3:8b`. Set the default local model in Settings.
- Or install [LM Studio](https://lmstudio.ai/), load a model, and start
  the local server (default `http://localhost:1234`).
- The Settings panel has Ollama URL and LM Studio URL fields — adjust
  if you've moved either off its default port.

## Model load failures

Local models fail to respond when:
- The selected model isn't actually loaded in Ollama / LM Studio.
- The model name in Settings doesn't match what's available (case
  matters; check `ollama list`).
- The model is too large for your VRAM and the runtime is silently
  falling back to CPU at unusable speeds.

If a local response comes back empty or scores low, the orchestrator
auto-escalates that single turn to Claude.

## "backend\\.venv is missing"

`Start.bat`'s install step didn't finish. Re-run `Start.bat`; the
script is safe to run multiple times.

## Sidecar fails to start

Open `%APPDATA%\iMakeAiTeams\sidecar.log` and read the last 50 lines.
Common causes:
- Port in use by another process.
- Antivirus blocking the PyInstaller-bundled `server.exe`.
- Missing Visual C++ redistributable.

## `npm run dev` exits immediately

`node_modules/` is incomplete. Delete it and re-run `Start.bat` (or
`npm install` directly).

## App opens to a blank window

Likely a renderer build error. Check the terminal where `Start.bat` is
running for stack traces. The Electron main process and sidecar
restart automatically when their source files change; the renderer
hot-reloads on save.

## Diagnostics export

When in doubt, Settings → Troubleshooting → Export diagnostics produces
a zip with logs, settings (with secrets redacted), and version
information. Attach that to a bug report.
