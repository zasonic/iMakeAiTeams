// desktop-ui/stores/appStore.ts — Zustand 5.x app store.
//
// Persists user preferences via localStorage (Zustand `persist` middleware).
// Runtime state (sidecar status, conversation streaming, error toasts) is
// kept in-memory and stripped from the persisted snapshot via `partialize`.

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

import type { SidecarStatus } from "../../desktop-shell/sidecar";

export type ActiveView =
  | "chat"
  | "agents"
  | "rag"
  | "memory"
  | "memory_review"
  | "prompts"
  | "mcp"
  | "security"
  | "settings"
  | "diagnostics"
  | "escalations";

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

// ── Power Mode (v3) ─────────────────────────────────────────────────────────

export type ExecutionStepKind =
  | "thinking"
  | "tool_call"
  | "file_write"
  | "shell"
  | "web"
  | "other";

export interface ExecutionStep {
  step_id: string;
  kind: ExecutionStepKind;
  title?: string;
  detail?: string;
  path?: string;
  preview?: string;
  command?: string;
  stdout?: string;
  stderr?: string;
  exit_code?: number;
  url?: string;
  summary?: string;
  args?: unknown;
  result?: unknown;
  bytes?: number;
  image_data?: string;   // base64 or data URL for inline rendering
  image_url?: string;    // URL to image file in container
  status: "running" | "done" | "error";
}

export interface ExecutionApproval {
  approval_id: string;
  summary: string;
  details: Record<string, unknown>;
  danger: "low" | "medium" | "high";
  expires_at: number;
}

export interface PowerModeRun {
  taskId: string;
  conversationId: string;
  startedAt: number;
  steps: ExecutionStep[];
  approvals: ExecutionApproval[];
  resultText: string;
  error: string;
  done: boolean;
}

export interface Escalation {
  id: string;
  conversation_id: string;
  triggered_at: string;
  trigger_type: string;
  trigger_detail: string;
  model_input?: string;
  proposed_action?: string | null;
  decision?: string | null;
  decided_at?: string | null;
}

export interface PendingWrite {
  id: string;
  conversation_id: string | null;
  write_type: string;
  content: string;
  contradicts_id: string | null;
  contradicts_content: string | null;
  proposed_at: string;
  decision: string | null;
  decided_at: string | null;
}

export interface DockerStatusSnapshot {
  wsl_installed: boolean;
  docker_installed: boolean;
  docker_running: boolean;
  openclaw_running: boolean;
  openclaw_healthy: boolean;
  gpu_available: boolean;
  platform: string;
  detail: string;
  last_error: string;
  gateway_url: string;
  workspace_dir: string;
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

  // Power Mode runtime
  powerModeRuns: Record<string, PowerModeRun>;
  dockerStatus: DockerStatusSnapshot | null;
  powerModeEnabled: boolean;

  // Phase 5: Wiser-Human escalation queue
  pendingEscalations: Escalation[];

  // Phase 5: MINJA-style memory write gate queue
  pendingMemoryWrites: PendingWrite[];

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

  // Power Mode actions
  setPowerModeEnabled: (on: boolean) => void;
  setDockerStatus: (s: DockerStatusSnapshot | null) => void;
  startPowerModeRun: (taskId: string, conversationId: string) => void;
  upsertPowerModeStep: (taskId: string, step: ExecutionStep) => void;
  addPowerModeApproval: (taskId: string, approval: ExecutionApproval) => void;
  resolvePowerModeApproval: (taskId: string, approvalId: string) => void;
  setPowerModeMessage: (taskId: string, text: string) => void;
  setPowerModeError: (taskId: string, error: string) => void;
  endPowerModeRun: (taskId: string) => void;

  // Escalation actions (Phase 5)
  setPendingEscalations: (list: Escalation[]) => void;
  addEscalation: (e: Escalation) => void;
  removeEscalation: (id: string) => void;

  // Memory write gate actions (Phase 5)
  setPendingMemoryWrites: (list: PendingWrite[]) => void;
  addPendingMemoryWrite: (w: PendingWrite) => void;
  removePendingMemoryWrite: (id: string) => void;
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
      powerModeRuns: {},
      dockerStatus: null,
      powerModeEnabled: false,
      pendingEscalations: [],
      pendingMemoryWrites: [],

      setActiveView: (v) => set({ activeView: v }),
      setStudioMode: (on) => set({ studioMode: on }),
      setHasCompletedFirstRun: (done) => set({ hasCompletedFirstRun: done }),
      setSidecarStatus: (s) => set({ sidecarStatus: s }),
      setServiceStatus: (s) => set({ serviceStatus: s }),
      pushToast: (msg) => {
        // crypto.randomUUID is on every browser Electron 33 ships, but fall
        // back to a longer random suffix on older runtimes just in case.
        const id =
          typeof crypto !== "undefined" && "randomUUID" in crypto
            ? crypto.randomUUID()
            : `${Date.now()}-${Math.random().toString(36).slice(2, 12)}`;
        set((state) => ({ toasts: [...state.toasts, { id, ...msg }] }));
        // Auto-dismiss so a misbehaving sidecar can't flood the UI with
        // service_unavailable toasts that pile up forever. Errors and
        // warnings stick around longer so the user has time to read them.
        const ms = msg.kind === "error" || msg.kind === "warn" ? 8000 : 4000;
        setTimeout(() => {
          set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) }));
        }, ms);
      },
      dismissToast: (id) =>
        set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),
      startChatStream: (conversationId) =>
        set({ activeChat: { conversationId, buffer: "", events: [] } }),
      appendChatToken: (token) =>
        set((state) => {
          if (!state.activeChat) return state;
          // Cap the streaming buffer so a long response doesn't pin the
          // whole transcript in memory and quadratic-copy it on every
          // token. The renderer only displays a window of recent text;
          // once we cross MAX, drop the head and keep the tail.
          const MAX = 1_000_000; // ~1 MiB of streamed text
          const KEEP = 500_000;
          const next = state.activeChat.buffer + token;
          const trimmed = next.length > MAX ? next.slice(next.length - KEEP) : next;
          return {
            activeChat: { ...state.activeChat, buffer: trimmed },
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

      // ── Power Mode actions ────────────────────────────────────────────
      setPowerModeEnabled: (on) => set({ powerModeEnabled: on }),
      setDockerStatus: (s) => set({ dockerStatus: s }),
      startPowerModeRun: (taskId, conversationId) =>
        set((state) => ({
          powerModeRuns: {
            ...state.powerModeRuns,
            [taskId]: {
              taskId,
              conversationId,
              startedAt: Date.now(),
              steps: [],
              approvals: [],
              resultText: "",
              error: "",
              done: false,
            },
          },
        })),
      upsertPowerModeStep: (taskId, step) =>
        set((state) => {
          const run = state.powerModeRuns[taskId];
          if (!run) return state;
          const existingIdx = run.steps.findIndex((s) => s.step_id === step.step_id);
          const nextSteps = existingIdx >= 0
            ? run.steps.map((s, i) => (i === existingIdx ? { ...s, ...step } : s))
            : [...run.steps, step];
          return {
            powerModeRuns: {
              ...state.powerModeRuns,
              [taskId]: { ...run, steps: nextSteps },
            },
          };
        }),
      addPowerModeApproval: (taskId, approval) =>
        set((state) => {
          const run = state.powerModeRuns[taskId];
          if (!run) return state;
          if (run.approvals.some((a) => a.approval_id === approval.approval_id)) {
            return state;
          }
          return {
            powerModeRuns: {
              ...state.powerModeRuns,
              [taskId]: { ...run, approvals: [...run.approvals, approval] },
            },
          };
        }),
      resolvePowerModeApproval: (taskId, approvalId) =>
        set((state) => {
          const run = state.powerModeRuns[taskId];
          if (!run) return state;
          return {
            powerModeRuns: {
              ...state.powerModeRuns,
              [taskId]: {
                ...run,
                approvals: run.approvals.filter((a) => a.approval_id !== approvalId),
              },
            },
          };
        }),
      setPowerModeMessage: (taskId, text) =>
        set((state) => {
          const run = state.powerModeRuns[taskId];
          if (!run) return state;
          return {
            powerModeRuns: {
              ...state.powerModeRuns,
              [taskId]: { ...run, resultText: (run.resultText || "") + text },
            },
          };
        }),
      setPowerModeError: (taskId, error) =>
        set((state) => {
          const run = state.powerModeRuns[taskId];
          if (!run) return state;
          return {
            powerModeRuns: {
              ...state.powerModeRuns,
              [taskId]: { ...run, error, done: true },
            },
          };
        }),
      // ── Escalation actions (Phase 5) ──────────────────────────────────
      setPendingEscalations: (list) => set({ pendingEscalations: list }),
      addEscalation: (e) =>
        set((state) => {
          if (state.pendingEscalations.some((p) => p.id === e.id)) {
            return state;
          }
          return { pendingEscalations: [...state.pendingEscalations, e] };
        }),
      removeEscalation: (id) =>
        set((state) => ({
          pendingEscalations: state.pendingEscalations.filter((p) => p.id !== id),
        })),

      // ── Memory write gate actions (Phase 5) ──────────────────────────
      setPendingMemoryWrites: (list) => set({ pendingMemoryWrites: list }),
      addPendingMemoryWrite: (w) =>
        set((state) => {
          if (state.pendingMemoryWrites.some((p) => p.id === w.id)) {
            return state;
          }
          return { pendingMemoryWrites: [...state.pendingMemoryWrites, w] };
        }),
      removePendingMemoryWrite: (id) =>
        set((state) => ({
          pendingMemoryWrites: state.pendingMemoryWrites.filter((p) => p.id !== id),
        })),

      endPowerModeRun: (taskId) => {
        set((state) => {
          const run = state.powerModeRuns[taskId];
          if (!run) return state;
          // Drop approvals that already expired so a finished run can't
          // leave stale prompts in state for the renderer to show.
          const now = Date.now();
          const liveApprovals = run.approvals.filter(
            (a) => a.expires_at >= now,
          );
          return {
            powerModeRuns: {
              ...state.powerModeRuns,
              [taskId]: { ...run, approvals: liveApprovals, done: true },
            },
          };
        });
        // Sweep completed runs older than 2h. Without this, powerModeRuns
        // grows unbounded across heavy sessions. Active runs are never
        // touched — the filter only removes done:true rows past the cutoff.
        const TTL_MS = 7_200_000;
        setTimeout(() => {
          set((state) => {
            const cutoff = Date.now() - TTL_MS;
            const next: Record<string, PowerModeRun> = {};
            for (const [id, run] of Object.entries(state.powerModeRuns)) {
              if (run.done && run.startedAt < cutoff) continue;
              next[id] = run;
            }
            return { powerModeRuns: next };
          });
        }, TTL_MS);
      },
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
