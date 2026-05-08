// desktop-shell/main.ts — Electron main process entrypoint.
//
// Lifecycle:
//   1. App ready → spawn sidecar (SidecarManager) → open BrowserWindow
//   2. Wire IPC: sidecar:get-info, sidecar:restart, dialog:*, shell:*, updater:*
//   3. On quit → POST /shutdown to sidecar → kill if grace period elapses
//
// Security: contextIsolation:true, nodeIntegration:false, sandbox:true.
// All network is 127.0.0.1 — see CSP in index.html.

import { app, BrowserWindow, dialog, ipcMain, Menu, session, shell } from "electron";
import { autoUpdater } from "electron-updater";
import { createWriteStream, existsSync, mkdirSync, statSync, renameSync, WriteStream } from "node:fs";
import { writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { fileURLToPath } from "node:url";

import { SidecarManager } from "./sidecar";

// Only meaningful in development — `resolveSpawnArgs` consults this to find
// the source-tree venv. In a packaged build the path resolves to a location
// inside the asar archive that doesn't exist on disk, so the SidecarManager
// only reads it from its `else (app.isPackaged)` branch.
const PROJECT_ROOT = app.isPackaged
  ? app.getAppPath()
  : fileURLToPath(new URL("../..", import.meta.url));

let mainWindow: BrowserWindow | null = null;
let sidecar: SidecarManager | null = null;
let mainLogStream: WriteStream | null = null;
let updaterTimer: NodeJS.Timeout | null = null;

const MAIN_LOG_MAX_BYTES = 10 * 1024 * 1024; // 10MB

function logToFile(text: string): void {
  try {
    mainLogStream?.write(text);
  } catch {
    /* ignore */
  }
}

// Per-handler call timestamps (epoch ms) for the IPC sliding-window rate limiter.
const _rateLimitWindows = new Map<string, number[]>();

function rateLimit(
  name: string,
  maxPerSecond: number,
  handler: (...args: unknown[]) => unknown,
): (...args: unknown[]) => unknown {
  return (...args: unknown[]) => {
    const now = Date.now();
    const windowStart = now - 1000;
    let calls = _rateLimitWindows.get(name);
    if (!calls) {
      calls = [];
      _rateLimitWindows.set(name, calls);
    }
    // Drop timestamps that have aged out of the 1s window.
    while (calls.length > 0 && calls[0] < windowStart) calls.shift();
    if (calls.length >= maxPerSecond) {
      const message = `rate limit exceeded: ${name}`;
      logToFile(`ipc rate limit: ${name} (${calls.length}/${maxPerSecond} in 1s)\n`);
      throw new Error(message);
    }
    calls.push(now);
    return handler(...args);
  };
}

function bootMainLog(userDataDir: string): void {
  if (!existsSync(userDataDir)) mkdirSync(userDataDir, { recursive: true });
  const path = join(userDataDir, "main.log");
  try {
    if (existsSync(path) && statSync(path).size > MAIN_LOG_MAX_BYTES) {
      try {
        renameSync(path, `${path}.1`);
      } catch {
        /* ignore */
      }
    }
  } catch {
    /* ignore */
  }
  mainLogStream = createWriteStream(path, { flags: "a" });
  logToFile(`\n=== main starting at ${new Date().toISOString()} ===\n`);
}

function wireSidecarAuthHeader(targetSession: Electron.Session): void {
  // Inject Authorization: Bearer <token> on every renderer request to the
  // sidecar so EventSource (which can't set headers) and stray fetch() calls
  // don't have to ship the token in URLs or log lines. The token lives in
  // the main process; the renderer never needs to see it.
  targetSession.webRequest.onBeforeSendHeaders(
    { urls: ["http://127.0.0.1:*/*"] },
    (details, callback) => {
      const info = sidecar?.getInfo();
      if (info) {
        try {
          if (new URL(details.url).port === String(info.port)) {
            details.requestHeaders["Authorization"] = `Bearer ${info.token}`;
          }
        } catch {
          // Malformed URL — pass through without injecting auth rather than
          // crashing the request listener.
        }
      }
      callback({ requestHeaders: details.requestHeaders });
    },
  );
}

async function createWindow(): Promise<void> {
  wireSidecarAuthHeader(session.defaultSession);

  mainWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 1024,
    minHeight: 660,
    backgroundColor: "#0a0a0c",
    show: false,
    autoHideMenuBar: true,
    webPreferences: {
      preload: join(__dirname, "../preload/index.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
      // Block the F12 / Ctrl+Shift+I shortcut in packaged builds so a
      // mis-click can't surface internals to end users. Devs running
      // `npm run dev` keep DevTools.
      devTools: !app.isPackaged,
    },
  });

  mainWindow.once("ready-to-show", () => {
    mainWindow?.show();
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:/.test(url)) {
      shell.openExternal(url).catch(() => {});
    }
    return { action: "deny" };
  });

  // Block in-window navigation away from the app shell. Without this,
  // a stray <a target="_self"> or `window.location = "https://evil"` —
  // whether from a renderer bug or XSS — could replace the app UI with
  // remote content. Allow only the dev-server URL and the packaged
  // file:// renderer; everything else opens externally instead.
  mainWindow.webContents.on("will-navigate", (event, url) => {
    const dev = process.env.ELECTRON_RENDERER_URL;
    if (dev && url.startsWith(dev)) return;
    if (url.startsWith("file://")) return;
    event.preventDefault();
    if (/^https?:\/\//i.test(url)) {
      shell.openExternal(url).catch(() => {});
    }
  });

  if (process.env.ELECTRON_RENDERER_URL) {
    await mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL);
  } else {
    await mainWindow.loadFile(join(__dirname, "../renderer/index.html"));
  }
}

function wireIpc(): void {
  ipcMain.handle("sidecar:get-info", () => sidecar?.getInfo() ?? null);
  ipcMain.handle(
    "sidecar:restart",
    rateLimit("sidecar:restart", 2, async () => {
      if (!sidecar) throw new Error("Sidecar manager not initialized");
      return sidecar.restart();
    }),
  );

  ipcMain.handle(
    "dialog:select-folder",
    rateLimit("dialog:select-folder", 4, async () => {
      if (!mainWindow) return null;
      const result = await dialog.showOpenDialog(mainWindow, {
        properties: ["openDirectory"],
      });
      if (result.canceled || result.filePaths.length === 0) return null;
      return result.filePaths[0];
    }),
  );

  ipcMain.handle(
    "dialog:select-files",
    rateLimit("dialog:select-files", 4, (async (_e, filters?: { name: string; extensions: string[] }[]) => {
      if (!mainWindow) return [];
      const result = await dialog.showOpenDialog(mainWindow, {
        properties: ["openFile", "multiSelections"],
        filters: filters ?? [
          {
            name: "Documents",
            extensions: [
              "txt", "md", "pdf", "py", "js", "json", "csv", "html", "css",
              "ts", "jsx", "tsx", "yaml", "yml", "toml", "xml", "sql", "sh",
              "bat", "ps1", "rs", "go", "java", "c", "cpp", "h", "rb",
            ],
          },
          { name: "All Files", extensions: ["*"] },
        ],
      });
      if (result.canceled) return [];
      return result.filePaths;
    }) as (...args: unknown[]) => unknown),
  );

  ipcMain.handle(
    "dialog:save-file",
    rateLimit("dialog:save-file", 4, (async (_e, { suggestedName, content }: { suggestedName: string; content: string }) => {
      if (!mainWindow) return { ok: false, error: "no window" };
      // Strip any path components from the renderer-supplied name so a
      // compromised renderer can't pre-fill the dialog with /etc/passwd or
      // C:\Windows\System32\... and trick the user into clicking Save.
      const safeName = basename(typeof suggestedName === "string" ? suggestedName : "");
      if (!safeName || safeName === "." || safeName === "..") {
        return { ok: false, error: "invalid name" };
      }
      const SAVE_FILE_MAX_BYTES = 50 * 1024 * 1024; // 50 MiB
      if (typeof content !== "string") {
        return { ok: false, error: "content too large" };
      }
      // Cap on the encoded byte length, not String.length, so non-ASCII
      // content can't sneak past the limit at ~2× its UTF-16 size.
      if (Buffer.byteLength(content, "utf-8") > SAVE_FILE_MAX_BYTES) {
        return { ok: false, error: "content too large" };
      }
      const result = await dialog.showSaveDialog(mainWindow, {
        defaultPath: safeName,
      });
      if (result.canceled || !result.filePath) return { ok: false, cancelled: true };
      try {
        await writeFile(result.filePath, content, "utf-8");
        return { ok: true, path: result.filePath };
      } catch (err) {
        return { ok: false, error: err instanceof Error ? err.message : String(err) };
      }
    }) as (...args: unknown[]) => unknown),
  );

  ipcMain.handle(
    "export:pdf",
    rateLimit("export:pdf", 4, (async (_e, payload: unknown) => {
      // Render the renderer-supplied HTML in a hidden BrowserWindow, then
      // hand the bytes off to a native Save dialog. printToPDF runs entirely
      // in-process — no headless Chrome spawn, no third-party dependency.
      if (!mainWindow) return { ok: false, error: "no window" };
      const data = (payload ?? {}) as { html?: unknown; suggestedName?: unknown };
      const html = typeof data.html === "string" ? data.html : "";
      const rawName =
        typeof data.suggestedName === "string" ? data.suggestedName : "";
      // Same defensive cleanup as dialog:save-file — strip any path
      // components and reject bare "."/".." so a compromised renderer can't
      // pre-fill the dialog with /etc/anything.
      const safeName = basename(rawName) || "conversation.pdf";
      if (safeName === "." || safeName === "..") {
        return { ok: false, error: "invalid name" };
      }
      // 25 MiB cap — a 25 MiB string is already an absurd HTML payload, and
      // the cap stops a runaway renderer from feeding us multi-GB documents.
      const HTML_MAX_BYTES = 25 * 1024 * 1024;
      if (Buffer.byteLength(html, "utf-8") > HTML_MAX_BYTES) {
        return { ok: false, error: "html too large" };
      }
      if (!html) return { ok: false, error: "empty html" };

      const win = new BrowserWindow({
        show: false,
        webPreferences: {
          // Hidden window only renders the HTML we just built — no preload,
          // no Node integration, no remote loading. Sandbox keeps it safe
          // even if the HTML contained a script tag.
          contextIsolation: true,
          nodeIntegration: false,
          sandbox: true,
          javascript: false,
        },
      });
      try {
        await win.loadURL(
          "data:text/html;charset=utf-8," + encodeURIComponent(html),
        );
        const pdfBytes = await win.webContents.printToPDF({
          printBackground: true,
          pageSize: "A4",
        });
        const result = await dialog.showSaveDialog(mainWindow, {
          defaultPath: safeName,
          filters: [{ name: "PDF", extensions: ["pdf"] }],
        });
        if (result.canceled || !result.filePath) {
          return { ok: false, cancelled: true };
        }
        await writeFile(result.filePath, pdfBytes);
        return { ok: true, path: result.filePath };
      } catch (err) {
        return {
          ok: false,
          error: err instanceof Error ? err.message : String(err),
        };
      } finally {
        if (!win.isDestroyed()) win.destroy();
      }
    }) as (...args: unknown[]) => unknown),
  );

  ipcMain.handle("shell:open-external", async (_e, url: string) => {
    if (typeof url !== "string") return;
    if (!/^https?:\/\//i.test(url)) return;
    await shell.openExternal(url);
  });

  ipcMain.handle("app:version", () => app.getVersion());
  ipcMain.handle("app:user-data-path", () => app.getPath("userData"));

  ipcMain.handle(
    "update:install-now",
    rateLimit("update:install-now", 1, async () => {
      // The user already confirmed by clicking "Restart now" in the
      // UpdateBanner — no native dialog here, just shut down cleanly
      // and let electron-updater swap the binaries. Stop the sidecar
      // BEFORE quitAndInstall: NSIS on Windows can't replace files that
      // are still open (server.exe), so an active sidecar turns the
      // install into a "file in use" failure. before-quit sees
      // sidecar=null afterwards and skips its own teardown.
      if (sidecar) {
        try {
          await sidecar.stop();
        } catch {
          /* best-effort */
        }
        sidecar = null;
      }
      autoUpdater.quitAndInstall(false, true);
      return { ok: true };
    }),
  );
}

interface SidecarHttpInfo {
  port: number;
  token: string;
}

async function fetchAutoUpdateEnabled(info: SidecarHttpInfo): Promise<boolean> {
  // Pulled in main rather than via wireSidecarAuthHeader: that hook only
  // augments the renderer session, not main-process fetch. Keep this
  // best-effort — a hung sidecar must not block update polling forever,
  // so a 2s timeout returns the spec-mandated `true` fallback instead.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 2_000);
  try {
    const res = await fetch(
      `http://127.0.0.1:${info.port}/api/settings/get?key=auto_update_enabled`,
      {
        headers: { Authorization: `Bearer ${info.token}` },
        signal: controller.signal,
      },
    );
    if (!res.ok) return true;
    const body = (await res.json()) as { value?: unknown };
    return body.value !== false;
  } catch {
    return true;
  } finally {
    clearTimeout(timer);
  }
}

// Constructed once and reused for every "update-available" payload. The
// publish target in electron-builder.yml is the source of truth; this
// constant must mirror it. Keep in sync if the publish section ever moves.
const RELEASE_NOTES_BASE = "https://github.com/zasonic/iMakeAiTeams/releases/tag";

async function wireAutoUpdater(): Promise<void> {
  if (!app.isPackaged) return;

  autoUpdater.autoDownload = true;
  autoUpdater.autoInstallOnAppQuit = true;

  autoUpdater.on("update-available", (info) => {
    mainWindow?.webContents.send("update:available", {
      version: info.version,
      notesUrl: `${RELEASE_NOTES_BASE}/v${info.version}`,
    });
  });
  autoUpdater.on("update-downloaded", (info) => {
    mainWindow?.webContents.send("update:downloaded", { version: info.version });
  });
  autoUpdater.on("error", (err) => {
    // Log only — surfacing this via a dialog would break the
    // non-blocking guarantee. The user simply doesn't see a banner
    // when the check fails, and the next 6h tick retries.
    logToFile(`autoUpdater error: ${err.message}\n`);
  });

  // Honour the user's auto_update_enabled toggle. The fetch is best-effort:
  // if the sidecar isn't responding yet (cold start, crash) we fall back to
  // enabled=true and let the renderer flip the toggle off on next launch.
  const info = sidecar?.getInfo();
  const enabled = info ? await fetchAutoUpdateEnabled(info) : true;
  if (!enabled) return;

  // Check on launch and every 6 hours. Hold the interval handle so we can
  // clear it on quit (otherwise it keeps a reference to autoUpdater alive
  // and prevents clean process exit on some Electron versions).
  autoUpdater.checkForUpdatesAndNotify().catch(() => {});
  updaterTimer = setInterval(() => {
    autoUpdater.checkForUpdatesAndNotify().catch(() => {});
  }, 6 * 60 * 60 * 1000);
}

async function bootSidecar(userDataDir: string): Promise<void> {
  sidecar = new SidecarManager(PROJECT_ROOT, userDataDir);

  sidecar.on("status", (status) => {
    mainWindow?.webContents.send("sidecar:status", status);
    logToFile(`sidecar status: ${JSON.stringify(status)}\n`);
  });

  try {
    await sidecar.start();
  } catch (err) {
    logToFile(`sidecar.start failed: ${err instanceof Error ? err.message : err}\n`);
    // Window opens anyway so the renderer can show the error UI with a
    // "Restart Backend" button. The status event has already fired.
  }
}

app.whenReady().then(async () => {
  // Single-instance lock so multiple launches reuse the existing window.
  const got = app.requestSingleInstanceLock();
  if (!got) {
    app.quit();
    return;
  }
  app.on("second-instance", () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  // Strip the default menu in production; keep DevTools accessible in dev.
  if (app.isPackaged) Menu.setApplicationMenu(null);

  const userDataDir = app.getPath("userData");
  bootMainLog(userDataDir);

  wireIpc();
  await createWindow();
  await bootSidecar(userDataDir);
  await wireAutoUpdater();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  // On macOS the app conventionally stays alive with no windows; the dock
  // icon's "activate" event will reopen one. Keep the sidecar and log stream
  // running so reopening doesn't land on a dead backend. before-quit handles
  // teardown when the user actually quits.
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", async (event) => {
  if (updaterTimer) {
    clearInterval(updaterTimer);
    updaterTimer = null;
  }
  if (sidecar) {
    event.preventDefault();
    try {
      await sidecar.stop();
    } catch {
      /* ignore */
    }
    sidecar = null;
    mainLogStream?.end();
    mainLogStream = null;
    app.exit(0);
    return;
  }
  // Sidecar already torn down (e.g. via update:install-now). Still flush
  // the main log so the final lines from this session aren't truncated.
  mainLogStream?.end();
  mainLogStream = null;
});
