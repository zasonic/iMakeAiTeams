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

## Security benchmarks (AgentDojo)

`.github/workflows/security-bench.yml` runs the four published AgentDojo
suites (workspace, slack, banking, travel) against the security stack on
every push to `main`, commits the per-suite `benchmarks/<suite>.json`
files plus a regenerated [BENCHMARKS.md](../BENCHMARKS.md), and fails the
build if any suite's ASR exceeds the ceiling configured in
[`benchmarks/thresholds.json`](../benchmarks/thresholds.json).

Local reproduction (Windows):

```
dev\run-bench.bat
```

Or one suite at a time:

```
pip install -r backend\requirements-bench.txt
python -m backend.tests.agentdojo.run_suites --suite workspace --output benchmarks\workspace.json
python build-scripts\generate_benchmarks_md.py
```

Bench-only deps live in `backend/requirements-bench.txt`. They are NEVER
imported at runtime — `backend/pytest.ini` excludes
`backend/tests/agentdojo/` from default collection, so a regular
`pytest tests/` run still works on machines that have not installed the
bench deps.

### Required GitHub Actions secrets

Add both secrets under **Settings → Secrets and variables → Actions** on
the GitHub repository:

| Secret | Purpose |
| --- | --- |
| `ANTHROPIC_API_KEY` | Used by the Reader pass and the Actor's Anthropic LLM during the bench. |
| `BENCH_PUSH_TOKEN`  | A fine-grained personal access token with `contents: write` on this repo. The workflow uses it to commit `benchmarks/*.json` + `BENCHMARKS.md` back to `main`. The default `GITHUB_TOKEN` is intentionally not enough — pushes from `GITHUB_TOKEN` cannot retrigger workflows, but the commit-back step uses `[skip ci]` so the same loop is broken either way; the deploy-key route is preferred only because it lets you scope the token to this repo. |

`BENCH_PUSH_TOKEN` can also be a deploy key — see
[GitHub's deploy-keys docs](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys#deploy-keys).

### Tightening the threshold

Edit `benchmarks/thresholds.json`. The `max_asr_pct` per suite is the
hard ceiling; the `baseline_asr_pct` is the Hackett et al. (ACL 2025)
monolithic reference, only used for the rendered table.
