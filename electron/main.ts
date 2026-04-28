// electron/main.ts — Electron main process entrypoint.
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
import { appendFileSync, existsSync, mkdirSync, statSync, renameSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { fileURLToPath } from "node:url";

import { SidecarManager } from "./sidecar";

const PROJECT_ROOT = fileURLToPath(new URL("../..", import.meta.url));

let mainWindow: BrowserWindow | null = null;
let sidecar: SidecarManager | null = null;
let mainLogPath: string | null = null;
let updaterTimer: NodeJS.Timeout | null = null;

const MAIN_LOG_MAX_BYTES = 10 * 1024 * 1024; // 10MB

function logToFile(text: string): void {
  // Synchronous append: a buffered WriteStream lost its tail when the
  // app exited via OS shutdown / kill on macOS (where window-all-closed
  // doesn't trigger before-quit and the stream's .end() never ran).
  // Volume here is tiny — a handful of writes per session — so the
  // sync cost is irrelevant.
  if (!mainLogPath) return;
  try {
    appendFileSync(mainLogPath, text);
  } catch {
    /* ignore */
  }
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
  mainLogPath = path;
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
      if (info && new URL(details.url).port === String(info.port)) {
        details.requestHeaders["Authorization"] = `Bearer ${info.token}`;
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
  ipcMain.handle("sidecar:restart", async () => {
    if (!sidecar) throw new Error("Sidecar manager not initialized");
    return sidecar.restart();
  });

  ipcMain.handle("dialog:select-folder", async () => {
    if (!mainWindow) return null;
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ["openDirectory"],
    });
    if (result.canceled || result.filePaths.length === 0) return null;
    return result.filePaths[0];
  });

  ipcMain.handle("dialog:select-files", async (_e, filters?: { name: string; extensions: string[] }[]) => {
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
  });

  ipcMain.handle(
    "dialog:save-file",
    async (_e, { suggestedName, content }: { suggestedName: string; content: string }) => {
      if (!mainWindow) return { ok: false, error: "no window" };
      // Strip any path components from the renderer-supplied name so a
      // compromised renderer can't pre-fill the dialog with /etc/passwd or
      // C:\Windows\System32\... and trick the user into clicking Save.
      const safeName = basename(typeof suggestedName === "string" ? suggestedName : "");
      if (!safeName || safeName === "." || safeName === "..") {
        return { ok: false, error: "invalid name" };
      }
      const SAVE_FILE_MAX_BYTES = 50 * 1024 * 1024; // 50 MiB
      if (typeof content !== "string" || content.length > SAVE_FILE_MAX_BYTES) {
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
    },
  );

  ipcMain.handle("shell:open-external", async (_e, url: string) => {
    if (typeof url !== "string") return;
    if (!/^https?:\/\//i.test(url)) return;
    await shell.openExternal(url);
  });

  ipcMain.handle("app:version", () => app.getVersion());
  ipcMain.handle("app:user-data-path", () => app.getPath("userData"));

  ipcMain.handle("updater:install", async () => {
    // Always confirm with the user before restarting + reinstalling.
    // Without this, a compromised renderer could call window.electronAPI
    // and force-quit the app on demand.
    if (!mainWindow) return { ok: false, error: "no window" };
    const choice = await dialog.showMessageBox(mainWindow, {
      type: "question",
      buttons: ["Restart and install", "Later"],
      defaultId: 0,
      cancelId: 1,
      title: "Install update",
      message: "Restart iMakeAiTeams to install the downloaded update?",
      detail: "Any unsaved work will be lost.",
    });
    if (choice.response !== 0) return { ok: false, cancelled: true };
    autoUpdater.quitAndInstall(false, true);
    return { ok: true };
  });
}

function wireAutoUpdater(): void {
  if (!app.isPackaged) return;

  autoUpdater.autoDownload = true;
  autoUpdater.autoInstallOnAppQuit = true;

  autoUpdater.on("update-available", (info) => {
    mainWindow?.webContents.send("updater:available", { version: info.version });
  });
  autoUpdater.on("update-downloaded", (info) => {
    mainWindow?.webContents.send("updater:downloaded", { version: info.version });
  });
  autoUpdater.on("error", (err) => {
    logToFile(`autoUpdater error: ${err.message}\n`);
  });

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
  wireAutoUpdater();

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
    app.exit(0);
  }
});
