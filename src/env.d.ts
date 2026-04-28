/// <reference types="vite/client" />

import type { SidecarStatus } from "../electron/sidecar";

export interface SidecarInfo {
  port: number;
  token: string;
}

export interface ElectronAPI {
  getSidecarInfo: () => Promise<SidecarInfo | null>;
  restartSidecar: () => Promise<SidecarInfo>;
  selectFolder: () => Promise<string | null>;
  selectWorkspaceFolder: () => Promise<string | null>;
  selectFiles: (filters?: { name: string; extensions: string[] }[]) => Promise<string[]>;
  saveFileDialog: (
    suggestedName: string,
    content: string,
  ) => Promise<{ ok: boolean; path?: string; cancelled?: boolean; error?: string }>;
  openExternal: (url: string) => Promise<void>;
  getAppVersion: () => Promise<string>;
  getUserDataPath: () => Promise<string>;
  onSidecarStatus: (handler: (status: SidecarStatus) => void) => () => void;
  onUpdateAvailable: (handler: (info: { version: string }) => void) => () => void;
  onUpdateDownloaded: (handler: (info: { version: string }) => void) => () => void;
  installUpdate: () => Promise<void>;
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}

export {};
