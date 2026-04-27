// electron/main.ts — Electron main process entrypoint.
//
// Lifecycle:
//   1. App ready → spawn sidecar (SidecarManager) → open BrowserWindow
//   2. Wire IPC: sidecar:get-info, sidecar:restart, dialog:*, shell:*, updater:*
//   3. On quit → POST /shutdown to sidecar → kill if grace period elapses
//
// Security: contextIsolation:true, nodeIntegration:false, sandbox:true.
// All network is 127.0.0.1 — see CSP in index.html.

import { app, BrowserWindow, dialog, ipcMain, Menu, shell } from "electron";
import { autoUpdater } from "electron-updater";
import { createWriteStream, existsSync, mkdirSync, statSync, renameSync, WriteStream } from "node:fs";
import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { SidecarManager } from "./sidecar";

const PROJECT_ROOT = fileURLToPath(new URL("../..", import.meta.url));

let mainWindow: BrowserWindow | null = null;
let sidecar: SidecarManager | null = null;
let mainLogStream: WriteStream | null = null;

const MAIN_LOG_MAX_BYTES = 10 * 1024 * 1024; // 10MB

function logToFile(text: string): void {
  try {
    mainLogStream?.write(text);
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
  mainLogStream = createWriteStream(path, { flags: "a" });
  logToFile(`\n=== main starting at ${new Date().toISOString()} ===\n`);
}

async function createWindow(): Promise<void> {
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
      const result = await dialog.showSaveDialog(mainWindow, {
        defaultPath: suggestedName,
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

  ipcMain.handle("updater:install", () => {
    autoUpdater.quitAndInstall(false, true);
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

  // Check on launch and every 6 hours.
  autoUpdater.checkForUpdatesAndNotify().catch(() => {});
  setInterval(() => {
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
  if (sidecar) {
    event.preventDefault();
    try {
      await sidecar.stop();
    } catch {
      /* ignore */
    }
    sidecar = null;
    mainLogStream?.end();
    app.exit(0);
  }
});
