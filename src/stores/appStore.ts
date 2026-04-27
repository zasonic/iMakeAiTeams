// src/stores/appStore.ts — Zustand 5.x app store.
//
// Persists user preferences via localStorage (Zustand `persist` middleware).
// Runtime state (sidecar status, conversation streaming, error toasts) is
// kept in-memory and stripped from the persisted snapshot via `partialize`.

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

import type { SidecarStatus } from "../../electron/sidecar";

export type ActiveView =
  | "chat"
  | "agents"
  | "rag"
  | "memory"
  | "prompts"
  | "mcp"
  | "security"
  | "settings"
  | "diagnostics";

export interface ToastMessage {
  id: string;
  kind: "info" | "warn" | "error" | "success";
  text: string;
}

export interface ChatStreamState {
  conversationId: string;
  buffer: string;
  events: { type: string; data: unknown; at: number }[];
}

export interface AppState {
  // Persisted user preferences
  activeView: ActiveView;
  studioMode: boolean;
  hasCompletedFirstRun: boolean;

  // Runtime (not persisted)
  sidecarStatus: SidecarStatus | null;
  toasts: ToastMessage[];
  activeChat: ChatStreamState | null;
  serviceStatus: Record<string, { ok: boolean; error?: string | null }>;

  // Actions
  setActiveView: (v: ActiveView) => void;
  setStudioMode: (on: boolean) => void;
  setHasCompletedFirstRun: (done: boolean) => void;
  setSidecarStatus: (s: SidecarStatus) => void;
  setServiceStatus: (s: Record<string, { ok: boolean; error?: string | null }>) => void;
  pushToast: (msg: Omit<ToastMessage, "id">) => void;
  dismissToast: (id: string) => void;
  startChatStream: (conversationId: string) => void;
  appendChatToken: (token: string) => void;
  appendChatEvent: (type: string, data: unknown) => void;
  endChatStream: () => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      activeView: "chat",
      studioMode: false,
      hasCompletedFirstRun: false,

      sidecarStatus: null,
      toasts: [],
      activeChat: null,
      serviceStatus: {},

      setActiveView: (v) => set({ activeView: v }),
      setStudioMode: (on) => set({ studioMode: on }),
      setHasCompletedFirstRun: (done) => set({ hasCompletedFirstRun: done }),
      setSidecarStatus: (s) => set({ sidecarStatus: s }),
      setServiceStatus: (s) => set({ serviceStatus: s }),
      pushToast: (msg) =>
        set((state) => ({
          toasts: [
            ...state.toasts,
            { id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`, ...msg },
          ],
        })),
      dismissToast: (id) =>
        set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),
      startChatStream: (conversationId) =>
        set({ activeChat: { conversationId, buffer: "", events: [] } }),
      appendChatToken: (token) =>
        set((state) => {
          if (!state.activeChat) return state;
          return {
            activeChat: { ...state.activeChat, buffer: state.activeChat.buffer + token },
          };
        }),
      appendChatEvent: (type, data) =>
        set((state) => {
          if (!state.activeChat) return state;
          return {
            activeChat: {
              ...state.activeChat,
              events: [...state.activeChat.events, { type, data, at: Date.now() }],
            },
          };
        }),
      endChatStream: () => set({ activeChat: null }),
    }),
    {
      name: "imakeaiteams-prefs",
      storage: createJSONStorage(() => localStorage),
      // Only persist user preferences, never runtime state.
      partialize: (state) => ({
        activeView: state.activeView,
        studioMode: state.studioMode,
        hasCompletedFirstRun: state.hasCompletedFirstRun,
      }),
    },
  ),
);
