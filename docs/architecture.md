# Architecture

Three processes, one user-facing app.

## Components

- **Electron main** (`desktop-shell/main.ts`) — opens the window,
  spawns the sidecar, brokers IPC, injects the bearer token on every
  renderer → sidecar request.
- **Renderer** (`desktop-ui/`) — React 19 + Zustand. Talks to the
  sidecar over REST and Server-Sent Events.
- **Python sidecar** (`backend/server.py`) — FastAPI, picks a random
  free port at startup and writes `PORT=<n>` to stdout. The brain of
  the app: chat orchestration, routing, memory, agents, MCP, Power
  Mode bridge.

## Process boundaries

1. Electron main launches; spawns the sidecar via
   `desktop-shell/sidecar.ts`. The sidecar prints its port + bearer
   token on stdout.
2. Main captures both, exposes them to the renderer over IPC
   (`sidecar:get-info`).
3. Renderer (`desktop-ui/api/client.ts`) fetches `http://127.0.0.1:<port>`
   for REST and opens an SSE stream for live updates.
4. Main injects `Authorization: Bearer <token>` on every renderer →
   sidecar request via `session.webRequest.onBeforeSendHeaders`.

## Backend services

- `services/chat_orchestrator.py` — the per-turn chat loop.
- `services/hub_router.py` — single boundary for worker selection;
  scores agents by skill match, falls back to Qwen3 /no_think for
  ambiguous routes.
- `services/security_engine.py` — quarantine, deterministic rule
  enforcement, risk ledger.
- `services/memory.py` — buffer, fact store, RAG retrieval.
- `services/governance.py` — per-agent tool/budget policy.
- `services/qwen_thinking.py` — Qwen3 hybrid /think + /no_think paths.
- `services/execution_bridge.py` — Power Mode dispatch to OpenClaw.

## Data layout

`backend/core/paths.py` is the single source of truth for user-data
paths:

- Settings JSON: `user_dir() / "settings.json"`
- SQLite DB: `user_dir() / "myai.db"`
- API key: OS keyring under service name `iMakeAiTeams`

`user_dir()` resolves to `%APPDATA%/iMakeAiTeams` on Windows,
`~/Library/Application Support/iMakeAiTeams` on macOS, and
`~/.local/share/iMakeAiTeams` on Linux.

## Schema

`backend/db.py` is the single source of truth for SQLite. All schema
changes go through `_MIGRATIONS` with the `schema_migrations` table
tracking applied versions. Vector tables use sqlite-vec with
`vec_documents` / `vec_memories` virtual tables and `vec_*_map`
mapping tables that join on opaque IDs to the source rows.

## Build

- `npm run dev` — electron-vite dev mode + sidecar in subprocess.
- `npm run build` — produces `out/main`, `out/preload`, `out/renderer`.
- `npm run build:sidecar` — PyInstaller bundles `backend/` into
  `branding/sidecar-bundle/` for `electron-builder.yml` `extraResources`.
- `npm run dist` — chains the above and runs electron-builder for an
  NSIS Windows installer.

## User-facing labels

`backend/core/labels.py` and `desktop-ui/i18n/en.json` map internal
names (table names, service names) to display strings. The internal
names stay frozen so existing user databases keep working; only the
display strings travel out to the UI.
