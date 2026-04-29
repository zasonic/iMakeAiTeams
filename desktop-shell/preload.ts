// desktop-shell/preload.ts — typed bridge between renderer (sandboxed) and main.
//
// Only the methods enumerated below are exposed; everything else (Node, fs,
// child_process) stays on the main process side. JSON-only IPC; never raw
// Node objects across the boundary.

import { contextBridge, ipcRenderer } from "electron";

import type { SidecarStatus } from "./sidecar";

export interface SidecarInfo {
  port: number;
  token: string;
}

export interface ElectronAPI {
  /** Returns null until the sidecar reports ready. */
  getSidecarInfo: () => Promise<SidecarInfo | null>;
  /** Force a clean restart of the sidecar (kill + respawn). */
  restartSidecar: () => Promise<SidecarInfo>;
  /** Show a native folder picker; returns the chosen absolute path or null. */
  selectFolder: () => Promise<string | null>;
  /**
   * Show a native folder picker for the Power Mode workspace folder. Returns
   * the chosen absolute path or null if the user cancelled. Reuses the same
   * Electron dialog handler as selectFolder() — the separate name lets the
   * renderer make the intent obvious in Settings → Power Mode.
   */
  selectWorkspaceFolder: () => Promise<string | null>;
  /** Show a native multi-file picker; returns absolute paths. */
  selectFiles: (filters?: { name: string; extensions: string[] }[]) => Promise<string[]>;
  /** Write `content` to a path chosen via a native save dialog. */
  saveFileDialog: (suggestedName: string, content: string) => Promise<{
    ok: boolean;
    path?: string;
    cancelled?: boolean;
    error?: string;
  }>;
  /** Open a URL in the user's default browser. http(s) only. */
  openExternal: (url: string) => Promise<void>;

  /** App version reported by Electron's `app.getVersion()`. */
  getAppVersion: () => Promise<string>;
  /** Path of the Electron userData dir (where logs/db/settings live). */
  getUserDataPath: () => Promise<string>;

  /** Subscribe to sidecar lifecycle changes; returns an unsubscribe fn. */
  onSidecarStatus: (handler: (status: SidecarStatus) => void) => () => void;
  /** Subscribe to "update available" notifications from electron-updater. */
  onUpdateAvailable: (handler: (info: { version: string }) => void) => () => void;
  /** Subscribe to "update downloaded" notifications. */
  onUpdateDownloaded: (handler: (info: { version: string }) => void) => () => void;
  /** Restart and apply a downloaded update. */
  installUpdate: () => Promise<void>;
}

const api: ElectronAPI = {
  getSidecarInfo: () => ipcRenderer.invoke("sidecar:get-info"),
  restartSidecar: () => ipcRenderer.invoke("sidecar:restart"),

  selectFolder: () => ipcRenderer.invoke("dialog:select-folder"),
  selectWorkspaceFolder: () => ipcRenderer.invoke("dialog:select-folder"),
  selectFiles: (filters) => ipcRenderer.invoke("dialog:select-files", filters),
  saveFileDialog: (suggestedName, content) =>
    ipcRenderer.invoke("dialog:save-file", { suggestedName, content }),
  openExternal: (url) => ipcRenderer.invoke("shell:open-external", url),

  getAppVersion: () => ipcRenderer.invoke("app:version"),
  getUserDataPath: () => ipcRenderer.invoke("app:user-data-path"),

  onSidecarStatus: (handler) => {
    const wrapped = (_e: Electron.IpcRendererEvent, status: SidecarStatus) => handler(status);
    ipcRenderer.on("sidecar:status", wrapped);
    return () => ipcRenderer.removeListener("sidecar:status", wrapped);
  },
  onUpdateAvailable: (handler) => {
    const wrapped = (_e: Electron.IpcRendererEvent, info: { version: string }) => handler(info);
    ipcRenderer.on("updater:available", wrapped);
    return () => ipcRenderer.removeListener("updater:available", wrapped);
  },
  onUpdateDownloaded: (handler) => {
    const wrapped = (_e: Electron.IpcRendererEvent, info: { version: string }) => handler(info);
    ipcRenderer.on("updater:downloaded", wrapped);
    return () => ipcRenderer.removeListener("updater:downloaded", wrapped);
  },
  installUpdate: () => ipcRenderer.invoke("updater:install"),
};

contextBridge.exposeInMainWorld("electronAPI", api);
