// desktop-ui/stores/appStore.ts — Zustand 5.x app store.
//
// Persists user preferences via localStorage (Zustand `persist` middleware).
// Runtime state (sidecar status, conversation streaming, error toasts) is
// kept in-memory and stripped from the persisted snapshot via `partialize`.

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

import type { Attachment, PromptTemplate } from "@/api/client";
import type { SidecarStatus } from "../../desktop-shell/sidecar";

export type ActiveView =
  | "chat"
  | "agents"
  | "rag"
  | "memory"
  | "memory_review"
  | "models"
  | "prompts"
  | "saved_prompts"
  | "mcp"
  | "security"
  | "safety"
  | "usage"
  | "settings"
  | "diagnostics"
  | "escalations";

export type ToastAction = "open_canary_alert";

export interface ToastMessage {
  id: string;
  kind: "info" | "warn" | "error" | "success";
  text: string;
  action?: ToastAction;
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

export interface CanaryAlert {
  model_id: string;
  mean_drift: number;
  drifted_prompts: string[];
}

export interface BundledDownloadState {
  status: "idle" | "downloading" | "complete" | "error";
  modelId: string;
  bytesDone: number;
  bytesTotal: number;
  error: string;
}

// PR 17: voice asset download progress. Two-track because the UI surfaces
// STT and TTS download progress separately even though they're streamed by
// one /assets/download call. Status is the shared rollup so the wizard can
// show a single progress bar when both tracks are in flight.
export interface VoiceAssetsState {
  status: "idle" | "downloading" | "complete" | "error";
  sttReady: boolean;
  ttsReady: boolean;
  sttBytesDone: number;
  sttBytesTotal: number;
  ttsBytesDone: number;
  ttsBytesTotal: number;
  error: string;
}

// PR 17: voice recording session state. ``isRecording`` is the source of
// truth for the mic button highlight; ``recordingStartedAt`` lets the
// recording-indicator render an elapsed-time counter without re-renders.
export interface VoiceRecordingState {
  isRecording: boolean;
  recordingStartedAt: number;
  isTranscribing: boolean;
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

  // Phase 5: Local-model behavior-drift canary alert (most recent only)
  canaryAlert: CanaryAlert | null;
  canaryAlertOpen: boolean;

  // Phase 8: Symphony-style high-stakes consensus voting indicator.
  // Flipped on by the high_stakes_voting_started chat_event and off by
  // high_stakes_voting_complete; drives the StatusBar pill.
  votingActive: boolean;

  // Phase 9: Bundled llama-server download progress, driven by the SSE
  // bundled_download_progress / _complete / _error events emitted by the
  // backend POST /api/system/bundled/download handler. Lives in the store
  // so the wizard can survive a re-mount without losing progress state.
  bundledDownload: BundledDownloadState;

  // Phase 10: silent auto-update banner state. `updateReady` is set when
  // electron-updater fires "update-downloaded" via IPC; the UpdateBanner
  // shows until the user clicks "Restart now" or "Later". Dismissal is
  // session-scoped — never persisted — so the banner reappears on the next
  // launch if the update is still pending.
  updateReady: { version: string } | null;
  updateBannerDismissed: boolean;

  // PR 8: per-conversation pending file attachments (ephemeral or
  // persisted-to-RAG). Keyed by conversation id so the chip strip can
  // re-render instantly on conversation switch without a network round
  // trip. The backend is the source of truth; the renderer refetches
  // on conversation switch via Attachments.list().
  pendingAttachments: Record<string, Attachment[]>;

  // PR 17: voice. ``voiceAssets`` mirrors the bundled-download pattern; the
  // SSE handlers in App.tsx patch it as voice_assets_progress / _complete /
  // _error events arrive. ``voiceRecording`` is in-process state for the
  // mic button — never persisted, never observable to the backend.
  voiceAssets: VoiceAssetsState;
  voiceRecording: VoiceRecordingState;

  // PR 18: user-saved prompt templates. Cached client-side so the slash
  // picker in ChatView can render synchronously without a round-trip on
  // every keystroke. The PromptLibraryPanel rehydrates this from the
  // backend on mount; mutations write through and refresh.
  promptTemplates: PromptTemplate[];

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

  // Canary alert actions (Phase 5)
  setCanaryAlert: (alert: CanaryAlert | null) => void;
  setCanaryAlertOpen: (open: boolean) => void;

  // Voting indicator action (Phase 8)
  setVotingActive: (on: boolean) => void;

  // Bundled-download actions (Phase 9)
  setBundledDownload: (s: BundledDownloadState) => void;
  patchBundledDownload: (patch: Partial<BundledDownloadState>) => void;

  // Auto-update banner actions (Phase 10)
  setUpdateReady: (v: { version: string } | null) => void;
  setUpdateBannerDismissed: (b: boolean) => void;

  // Pending attachment actions (PR 8)
  setPendingAttachments: (conversationId: string, attachments: Attachment[]) => void;
  addPendingAttachment: (conversationId: string, a: Attachment) => void;
  removePendingAttachment: (conversationId: string, id: string) => void;

  // Voice actions (PR 17)
  setVoiceAssets: (s: VoiceAssetsState) => void;
  patchVoiceAssets: (patch: Partial<VoiceAssetsState>) => void;
  setVoiceRecording: (s: VoiceRecordingState) => void;
  patchVoiceRecording: (patch: Partial<VoiceRecordingState>) => void;

  // Prompt template actions (PR 18)
  setPromptTemplates: (rows: PromptTemplate[]) => void;
  upsertPromptTemplate: (t: PromptTemplate) => void;
  removePromptTemplate: (id: string) => void;
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
      canaryAlert: null,
      canaryAlertOpen: false,
      votingActive: false,
      bundledDownload: {
        status: "idle",
        modelId: "",
        bytesDone: 0,
        bytesTotal: 0,
        error: "",
      },
      updateReady: null,
      updateBannerDismissed: false,
      pendingAttachments: {},
      voiceAssets: {
        status: "idle",
        sttReady: false,
        ttsReady: false,
        sttBytesDone: 0,
        sttBytesTotal: 0,
        ttsBytesDone: 0,
        ttsBytesTotal: 0,
        error: "",
      },
      voiceRecording: {
        isRecording: false,
        recordingStartedAt: 0,
        isTranscribing: false,
      },
      promptTemplates: [],

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

      // ── Canary alert actions (Phase 5) ───────────────────────────────
      // Setting a fresh alert collapses the modal back to "closed" so we
      // never auto-open over the user — the toast click is the explicit
      // request to view details. Clearing the alert also closes the modal.
      setCanaryAlert: (alert) =>
        set({ canaryAlert: alert, canaryAlertOpen: false }),
      setCanaryAlertOpen: (open) => set({ canaryAlertOpen: open }),

      // ── Voting indicator action (Phase 8) ────────────────────────────
      setVotingActive: (on) => set({ votingActive: on }),

      // ── Bundled download (Phase 9) ────────────────────────────────────
      setBundledDownload: (s) => set({ bundledDownload: s }),
      patchBundledDownload: (patch) =>
        set((state) => ({ bundledDownload: { ...state.bundledDownload, ...patch } })),

      // ── Auto-update banner (Phase 10) ────────────────────────────────
      // Setting a fresh updateReady payload also resets the dismissed flag so
      // a brand-new download surfaces the banner even if the user dismissed
      // an earlier one in the same session.
      setUpdateReady: (v) =>
        set((state) => ({
          updateReady: v,
          updateBannerDismissed: v ? false : state.updateBannerDismissed,
        })),
      setUpdateBannerDismissed: (b) => set({ updateBannerDismissed: b }),

      // ── Pending attachments (PR 8) ───────────────────────────────────
      setPendingAttachments: (conversationId, attachments) =>
        set((state) => ({
          pendingAttachments: {
            ...state.pendingAttachments,
            [conversationId]: attachments,
          },
        })),
      addPendingAttachment: (conversationId, a) =>
        set((state) => {
          const existing = state.pendingAttachments[conversationId] ?? [];
          if (existing.some((x) => x.id === a.id)) return state;
          return {
            pendingAttachments: {
              ...state.pendingAttachments,
              [conversationId]: [...existing, a],
            },
          };
        }),
      removePendingAttachment: (conversationId, id) =>
        set((state) => {
          const existing = state.pendingAttachments[conversationId];
          if (!existing) return state;
          return {
            pendingAttachments: {
              ...state.pendingAttachments,
              [conversationId]: existing.filter((x) => x.id !== id),
            },
          };
        }),

      // ── Voice (PR 17) ────────────────────────────────────────────────
      setVoiceAssets: (s) => set({ voiceAssets: s }),
      patchVoiceAssets: (patch) =>
        set((state) => ({ voiceAssets: { ...state.voiceAssets, ...patch } })),
      setVoiceRecording: (s) => set({ voiceRecording: s }),
      patchVoiceRecording: (patch) =>
        set((state) => ({ voiceRecording: { ...state.voiceRecording, ...patch } })),

      // ── Prompt templates (PR 18) ─────────────────────────────────────
      setPromptTemplates: (rows) => set({ promptTemplates: rows }),
      upsertPromptTemplate: (t) =>
        set((state) => {
          const idx = state.promptTemplates.findIndex((p) => p.id === t.id);
          if (idx < 0) {
            return { promptTemplates: [...state.promptTemplates, t] };
          }
          const next = state.promptTemplates.slice();
          next[idx] = t;
          return { promptTemplates: next };
        }),
      removePromptTemplate: (id) =>
        set((state) => ({
          promptTemplates: state.promptTemplates.filter((p) => p.id !== id),
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
