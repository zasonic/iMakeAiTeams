# backend (Python FastAPI sidecar)

The brain: chat orchestration, agents, RAG, MCP, security.
Spawned by Electron on a random localhost port with a per-launch bearer
token (see `../desktop-shell/sidecar.ts`).

## Layout

| Path | Role |
|---|---|
| `server.py` | FastAPI app + lifespan + auth middleware |
| `db.py`, `models.py` | SQLite schema and ORM-ish helpers |
| `events_sse.py` | Server-Sent Events fan-out |
| `core/` | Paths, settings, first-run, event bus, worker thread |
| `core/paths.py` | Single source of truth for user-data locations |
| `routes/` | HTTP endpoints (one file per concern: chat, agents, rag, ...) |
| `services/` | Domain logic: orchestrator, router, memory, agents, MCP, security |
| `templates/` | Jinja2 templates (Caddyfile, docker-compose) |
| `pyinstaller.spec` | Frozen-binary build spec for shipping |

## Running

In dev, Electron launches `server.py` with `--token <uuid> --user-data <path>`.
Run it standalone for backend work:

```
cd backend
.venv\Scripts\activate              # or: source .venv/bin/activate
python server.py --token dev --user-data .scratch
```

Then hit `http://127.0.0.1:<port>/healthz` (port is printed to stdout as
`PORT=<n>` so the parent can capture it).

## State

All persistent state lives outside this folder, in
`core/paths.py:user_dir()` (`%APPDATA%/MyAIAgentHub` on Windows). This
folder is the application code; it never writes inside itself.
