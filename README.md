# iMakeAiTeams

A local-first desktop app where Claude and local models work as a coordinated
team. Electron shell + React UI + Python (FastAPI) sidecar.

You bring an Anthropic API key (and optionally a local model via Ollama or
LM Studio); the app routes each message to whichever model fits best, runs
multi-agent workflows, and persists everything to your machine — no SaaS
backend, no telemetry, no logins.

---

## Features

### Smart routing — pay Claude only for the hard parts

A small classifier runs on your **local model** (free) and decides where each
message goes:

- **Simple** (greetings, summarization, formatting) → local model
- **Medium** (analysis, code) → local if 13B+ available, otherwise Claude
- **Complex** (planning, judgment, creativity) → Claude

The router is **uncertainty-aware**: low-confidence local routes auto-escalate
to Claude. You can override anytime with `@claude` or `@local` in your message.

### Multi-agent teams

Six built-in agents — **coordinator, researcher, analyst, writer, coder,
reviewer** — each with its own system prompt, model preference, and skill
declarations. Compose them into teams; the **HubRouter** picks the right agent
for each task using deterministic skill matching, falling back to LLM-based
routing when the match is ambiguous.

Create your own agents from the **Agents** panel: custom prompt, model, skill
list, token budget. Each agent gets a **Theory of Mind** block describing what
it sees, what it can output, and where it fits in the team.

### Power Mode (sandboxed code execution)

Optional. When enabled, the assistant can delegate execution to **OpenClaw**
(a Docker-based sandbox): planning steps, tool calls, file reads/writes, shell
commands, and web actions stream back into the chat as inline execution cards.
Every potentially destructive proposal pauses for your approval (auto-denies
after 60 seconds). The whole feature is additive — if you don't enable it
you'll never see Docker.

### Three-tier memory

- **Short-term:** in-session conversation buffer
- **Working:** session facts auto-extracted by the local model and stored in SQLite ($0)
- **Long-term:** RAG index + semantic memory entries

A **trust scanner** screens content before it lands in long-term memory; flagged
items go to a pending-review queue you approve from the UI.

### RAG & semantic search (optional)

Index folders of documents; the assistant searches them for context with hybrid
**BM25 + semantic** retrieval. Per-source caps prevent any single document from
flooding context.

> RAG is opt-in. The default sidecar is the **lite** bundle (~80 MB) and skips
> the heavy ML stack (torch, sentence-transformers, chromadb). Enable it from
> the **Settings → Semantic search** panel and the app installs the extras on
> demand. Without them, BM25 keyword search still works.

### MCP (Model Context Protocol) integration

Drop MCP server manifests into the configured directory; iMakeAiTeams
auto-discovers tools, validates schemas, and surfaces them in the **MCP**
panel. Per-server credentials live in the OS keyring (Windows Credential
Manager / macOS Keychain / Linux Secret Service). Hot-reload picks up changes
without restart.

### Prompt library

Every system prompt the app uses is stored as a **versioned record** in
SQLite. Built-in prompts are read-only; duplicate to customize. Roll back to
any prior version. Agents reference prompts by name so you can A/B test
without code changes.

### Security engine (five layers, all deterministic)

1. **Context quarantine** — RAG chunks are tagged with provenance and capped
   per source so a poisoned document can't dominate the prompt.
2. **Risk ledger** — every tool/workflow call accrues calibrated risk;
   crossing a threshold aborts the chain.
3. **Memory firewall** — TTL enforcement, source attestation, structural
   validation, and automatic decay on stored memories.
4. **Skill scanner** — static regex over prompt templates and skill manifests
   to catch supply-chain injection patterns.
5. **Rule engine** — sub-millisecond pattern matching for role reassignment,
   base64/unicode smuggling, etc. — runs on every assembled context.

Plus a **prompt-injection scanner** (Meta PromptGuard 2 via LlamaFirewall) on
user input and ingested documents; **rate limiting** (10 chats/min, 120 API
calls/min); per-agent **governance** (tool allowlists, token caps, forbidden
patterns); and an append-only **audit log** of all lifecycle events.

### Goal decomposition + extended-reasoning visibility

Complex multi-part requests are automatically split into sequential steps
shown inline as a checklist. Claude's extended thinking is rendered as a
collapsible timeline with routing decisions, memory recalled, and reasoning
steps — so you can see *why* the assistant did what it did.

### First-class diagnostics

The **Diagnostics** panel shows: sidecar status, port, latency, token usage,
RAG index health, MCP server reachability, local-model availability, and
per-component health checks. The **Status Bar** keeps a colored dot visible
while you work.

### Auto-update

Built on `electron-updater` (GitHub releases). Checks on launch and hourly;
downloads in the background; installs on quit. Publisher-signed; the updater
verifies the certificate against `electron-builder.yml` to block MITM swaps.

---

## What's in this folder

| Path | What it is |
|---|---|
| `desktop-ui/` | React renderer — the windows, panels, and chat UI |
| `desktop-shell/` | Electron main + preload — desktop app host |
| `backend/` | Python FastAPI sidecar — chat, agents, memory, RAG, MCP, security |
| `branding/` | App icon + (gitignored) staged sidecar bundle |
| `build-scripts/` | Helpers invoked by `npm run` (sidecar build, packaging) |
| `archive/legacy-v5/` | Historical snapshot of the predecessor app. Not used at runtime. |
| `1-install.bat` / `.ps1` | First-time Windows setup (Node + Python + deps) |
| `2-run-dev.bat` | Start the app with hot-reload |
| `3-build-installer.bat` | Build the Windows NSIS installer |
| `README.md` | (You are here) |
| `QUICKSTART.md` | 5-minute install → run walkthrough + troubleshooting |
| `CONTRIBUTING.md` | Dev-from-source flow (any OS) |

---

## Prerequisites

| Requirement | Required? | Notes |
|---|---|---|
| **Windows 10/11** | For the one-click installer flow | Mac/Linux work via `CONTRIBUTING.md` |
| **Node.js 20+** | Yes (auto-installed by `1-install.bat`) | LTS recommended |
| **Python 3.12+** | Yes (auto-installed by `1-install.bat`) | venv created at `backend/.venv` |
| **Anthropic API key** | Yes for Claude features | Local-only mode possible if you have a strong local model |
| **Ollama or LM Studio** | Optional | Enables the routing classifier and local-model responses |
| **Docker / WSL2** | Optional | Only needed for Power Mode (OpenClaw) |
| **Semantic search ML stack** | Optional | Enabled on demand from Settings; lite bundle skips it by default |

---

## Quickstart (Windows)

```text
1. Double-click  1-install.bat        (one-time; installs Node, Python, npm + pip deps)
2. Double-click  2-run-dev.bat        (starts Electron + React + sidecar with hot-reload)
3. In the app    Settings → paste your Anthropic API key
```

To produce a distributable installer: double-click `3-build-installer.bat`.
The result lands in `dist/iMakeAiTeams-Setup-<version>.exe`. Test it on a
clean VM (no Python, no Node) — the installer bundles everything the end
user needs.

See [`QUICKSTART.md`](QUICKSTART.md) for a detailed walkthrough and
troubleshooting, or [`CONTRIBUTING.md`](CONTRIBUTING.md) for the
dev-from-source flow on Mac/Linux.

---

## First-launch checklist

Run after `2-run-dev.bat` opens the app window:

1. **Settings → API → Anthropic API key.** Paste your key. It's saved to the
   OS keyring under service name `iMakeAiTeams`, not to disk.
2. **(Optional) Settings → Local model.** If you've installed Ollama
   (`ollama pull llama3:8b`) or LM Studio, set the URL and default model.
   The router will start using it for cheap routing decisions.
3. **(Optional) RAG → Index a folder.** Pick a directory of documents; the
   first index downloads the embedding model (~90 MB). Subsequent indexes
   are incremental.
4. **(Optional) MCP → Add server.** Point at a folder containing an
   `mcp.json` manifest.
5. **(Optional) Power Mode.** Toggle in **Settings**; the app checks for
   Docker and offers to start the OpenClaw container.

You're ready.

---

## Architecture in one paragraph

The Electron shell (`desktop-shell/`) owns the window and spawns the Python
sidecar on a **random localhost port** with a **per-launch bearer token**.
The React renderer (`desktop-ui/`) talks to the sidecar over HTTP +
Server-Sent Events; all traffic is `127.0.0.1`. The sidecar (`backend/`)
classifies each message, routes it to Claude or a local model, runs agent
loops, manages memory, queries RAG/MCP, applies security checks, and
persists state to your OS user-data directory.

```
┌──────────────────────┐    HTTP + SSE        ┌───────────────────────┐
│ Electron + React UI  │◄──── 127.0.0.1 ────► │ FastAPI sidecar       │
│ (desktop-shell +     │   (random port,      │ - router              │
│  desktop-ui)         │    bearer token)     │ - chat orchestrator   │
└──────────┬───────────┘                      │ - agents + memory     │
           │                                  │ - RAG / MCP           │
           │ IPC                              │ - security engine     │
           ▼                                  └─────────┬─────────────┘
   OS dialogs, keyring,                                 │
   auto-update, file I/O                                ▼
                                          Anthropic API + local model
                                          (Ollama / LM Studio)
                                          + optional OpenClaw container
```

---

## Where things live at runtime

| What | Path (Windows) | macOS | Linux |
|---|---|---|---|
| Settings + database | `%APPDATA%\MyAIAgentHub\` | `~/Library/Application Support/MyAIAgentHub/` | `~/.local/share/MyAIAgentHub/` |
| Conversation DB | `myai.db` (in the above) | same | same |
| Main process log | `main.log` (10 MB rotation) | same | same |
| Sidecar log | `sidecar.log` | same | same |
| API key | Windows Credential Manager | macOS Keychain | Secret Service |
| OpenClaw compose / Caddyfile | `MyAIAgentHub/openclaw/` | same | same |

---

## Security model (defaults)

- **Sandboxed renderer.** `contextIsolation: true`, `nodeIntegration: false`,
  `sandbox: true`. The renderer can't touch the filesystem directly.
- **Local-only network.** All sidecar traffic is `127.0.0.1`; CSP enforces
  the same. Navigation away from the app shell is blocked.
- **Per-launch token.** The sidecar requires a bearer token rotated on every
  launch; the renderer never sees it (Electron injects it via
  `webRequest.onBeforeSendHeaders`).
- **OS-keyring secrets.** API keys and MCP credentials live in the OS
  keyring, never in `settings.json` or env vars.
- **Five-layer prompt-injection defense** (see Features → Security engine).
- **Sidecar firewall.** When Power Mode is on, OpenClaw is fronted by an
  auth-gated Caddy gateway with a `/healthz` probe.

---

## Building from source

If you want to develop or build the installer yourself:

```bash
npm install
cd backend && python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt && cd ..
npm run dev          # hot-reload dev
npm run typecheck    # tsc on desktop-shell + desktop-ui
npm run build        # electron-vite production build
npm run build:sidecar  # PyInstaller bundles backend → branding/sidecar-bundle/
npm run dist         # full installer (Windows only)
```

Full developer guide in [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Troubleshooting

A focused troubleshooting guide lives in
[`QUICKSTART.md`](QUICKSTART.md#troubleshooting). The common ones:

- `backend\.venv is missing` → re-run `1-install.bat` (idempotent).
- Blank window on launch → check the terminal where `2-run-dev.bat` is
  running for renderer build errors.
- Sidecar fails to start → tail `%APPDATA%\MyAIAgentHub\sidecar.log`.
- "Semantic search unavailable" → enable RAG in Settings; the app
  will install the optional ML stack on demand.

---

## Licence

UNLICENSED — see [`package.json`](package.json). Use is governed separately;
contact the maintainer.
