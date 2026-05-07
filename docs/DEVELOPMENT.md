# Development

Setup, daily workflow, and conventions.

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
   # Windows: .venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   cd ..
   ```

Windows users can run `Start.bat` instead — it does both of the above.

## Layout

```
desktop-ui/        React renderer (Vite, Tailwind)
desktop-shell/     Electron main + preload + sidecar manager
backend/           Python FastAPI sidecar (the brain)
branding/          App icon + staged sidecar bundle
build-scripts/     npm-script helpers
dev/               Developer entry scripts (install.ps1, build-installer.bat)
```

Entry points:
- Renderer: `desktop-ui/main.tsx` → `desktop-ui/App.tsx`
- Electron main: `desktop-shell/main.ts`
- Sidecar: `backend/server.py`

## Daily dev loop

```
npm run dev           # electron-vite dev + sidecar; renderer hot-reloads
npm run typecheck     # tsc on both desktop-shell and desktop-ui
npm run test:frontend # vitest run
cd backend && python -m pytest -q
```

The renderer hot-reloads on save. The Electron main process and
sidecar restart automatically when their source files change.

## Building

```
npm run build:sidecar    # PyInstaller → branding/sidecar-bundle/
npm run build            # electron-vite production build into out/
npm run dist             # build:sidecar + build + electron-builder NSIS
```

On Windows, `dev\build-installer.bat` chains all three and produces
`dist/iMakeAiTeams-Setup-<version>.exe`. Test the installer on a clean
VM (no Python, no Node).

## Code style

- TypeScript: strict mode; no `any` unless commented why.
- Python: type hints on all public functions; ruff-clean.
- Comments explain WHY, not WHAT.
- Comments rot when the code moves; prefer good names over comments.

## Pull requests

Push branches as `claude/<short-description>` (or your own prefix).
One concern per PR; commit messages in the form `<area>: <summary>`.
CI runs `npm run typecheck`, `npm run test:frontend`, and the backend
pytest suite.

## Schema and userData invariants

These are hard rules — breaking any of them breaks existing user
installs:

- `backend/core/paths.py` paths must not be renamed.
- All schema changes go through `_MIGRATIONS` in `backend/db.py` with
  a new version string and the `schema_migrations` table.
- Settings keys in `backend/core/settings.py` stay frozen.
- `electron-builder.yml` `extraResources` must keep resolving.
- `electron.vite.config.ts` `@/` alias must keep resolving to
  `desktop-ui/`.
- `backend/pyinstaller.spec` `collect_submodules("services" | "routes"
  | "core")` must keep finding everything.
