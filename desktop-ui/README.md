# desktop-ui (React renderer)

The React UI that runs inside the Electron BrowserWindow.

- Entry: `main.tsx` -> `App.tsx`
- State: `stores/appStore.ts` (Zustand)
- API client: `api/client.ts` (REST), `api/sse.ts` (Server-Sent Events)
- Components: `components/*` (one panel/view per file)
- Path alias: `@/*` -> this folder (configured in `electron.vite.config.ts`)

This folder runs in a sandboxed renderer with `contextIsolation: true`.
All file I/O, dialogs, and OS access go through `window.electronAPI`,
defined in `../desktop-shell/preload.ts` and typed in `env.d.ts`.

The backend URL is not hardcoded — it comes from
`window.electronAPI.getSidecarInfo()` because the sidecar picks a random
localhost port at launch.
