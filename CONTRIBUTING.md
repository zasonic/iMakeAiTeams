# Contributing

Developer setup and build flow.

## Layout

```
desktop-ui/        React renderer (Vite, Tailwind)
desktop-shell/     Electron main + preload + sidecar manager
backend/           Python FastAPI sidecar (the brain)
branding/          App icon + staged sidecar bundle
build-scripts/     npm-script helpers
```

Entry points:
- Renderer: `desktop-ui/main.tsx` -> `desktop-ui/App.tsx`
- Electron main: `desktop-shell/main.ts`
- Sidecar: `backend/server.py`

## First-time setup (any OS)

1. Install **Node 20+** and **Python 3.12+**.
2. Install JS deps:
   ```
   npm install
   ```
3. Create the Python venv and install backend deps:
   ```
   cd backend
   python -m venv .venv
   # Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   cd ..
   ```

(Windows users: `1-install.bat` does both of the above for you.)

## Daily dev loop

```
npm run dev          # starts electron-vite + sidecar, hot-reloads renderer
npm run typecheck    # tsc on both desktop-shell and desktop-ui
```

The renderer hot-reloads on save. The Electron main process and sidecar
restart automatically when their source files change.

## Building

```
npm run build:sidecar    # PyInstaller bundles backend/ into resources/
npm run build            # electron-vite production build into out/
npm run dist             # npm run build + electron-builder NSIS (Windows only)
```

On Windows, `3-build-installer.bat` chains all three and produces
`dist/iMakeAiTeams-Setup-<version>.exe`.

## How the pieces talk to each other

1. Electron main (`desktop-shell/main.ts`) launches; spawns the sidecar via
   `desktop-shell/sidecar.ts`. The sidecar picks a random free port and writes
   `PORT=<n>` to stdout.
2. Main captures the port + an in-memory bearer token, exposes both to the
   renderer over IPC (`sidecar:get-info`).
3. Renderer (`desktop-ui/api/client.ts`) fetches `http://127.0.0.1:<port>` for
   REST and opens a Server-Sent-Events stream for live updates.
4. Main injects `Authorization: Bearer <token>` on every renderer -> sidecar
   request via `session.webRequest.onBeforeSendHeaders`.

## Where state lives

`backend/core/paths.py` is the single source of truth for user-data paths:

- Settings JSON: `user_dir() / "settings.json"`
- SQLite DB: `user_dir() / "myai.db"`
- API key: OS keyring under service name `iMakeAiTeams`

`user_dir()` resolves to `%APPDATA%/MyAIAgentHub` on Windows,
`~/Library/Application Support/MyAIAgentHub` on macOS,
`~/.local/share/MyAIAgentHub` on Linux.

## Code style

- TypeScript: strict mode; no `any` unless commented why.
- Python: type hints on all public functions; ruff-clean.
- Comments explain WHY, not WHAT. Avoid restating what the code does.

## Pull requests

Push branches as `claude/<short-description>` (or your own prefix). One concern
per PR; keep commit messages in the form `<area>: <summary>`. The CI runs
`npm run typecheck` and a backend smoke harness.
