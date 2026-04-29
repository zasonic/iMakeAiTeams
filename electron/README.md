# desktop-shell (Electron main + preload)

The Electron host that owns the window and talks to the OS.

- `main.ts` — main process: window lifecycle, IPC handlers, auto-updater,
  CSP/navigation lockdown, log file rotation.
- `preload.ts` — bridge exposed to the renderer as `window.electronAPI`.
  Mirrors the typed surface declared in `../desktop-ui/env.d.ts`.
- `sidecar.ts` — spawns and supervises the Python FastAPI sidecar
  (`../backend/server.py` in dev, `resources/backend/server[.exe]` when
  packaged). Picks a random free port, generates a per-launch bearer token,
  and surfaces status events to the renderer.

Security posture: `contextIsolation: true`, `nodeIntegration: false`,
`sandbox: true`. All network is 127.0.0.1; CSP enforced in `index.html`.
Navigation away from the app shell is blocked by `will-navigate`.
