import { useEffect } from "react";

import { Escalation, Settings, System, resetSidecarInfo } from "@/api/client";
import { subscribeEvents } from "@/api/sse";
import { AgentPanel } from "@/components/AgentPanel";
import { ChatView } from "@/components/ChatView";
import { DiagnosticsPanel } from "@/components/DiagnosticsPanel";
import { EscalationModal } from "@/components/EscalationModal";
import { EscalationPanel } from "@/components/EscalationPanel";
import { FirstRunWizard } from "@/components/FirstRunWizard";
import { McpPanel } from "@/components/McpPanel";
import { MemoryPanel } from "@/components/MemoryPanel";
import { PromptPanel } from "@/components/PromptPanel";
import { RagPanel } from "@/components/RagPanel";
import { SecurityPanel } from "@/components/SecurityPanel";
import { SettingsPanel } from "@/components/SettingsPanel";
import { Sidebar } from "@/components/Sidebar";
import { StatusBar } from "@/components/StatusBar";
import {
  useAppStore,
  type DockerStatusSnapshot,
  type Escalation as EscalationItem,
  type ExecutionStep,
  type ExecutionStepKind,
} from "@/stores/appStore";

export function App() {
  const view = useAppStore((s) => s.activeView);
  const sidecarStatus = useAppStore((s) => s.sidecarStatus);
  const setSidecarStatus = useAppStore((s) => s.setSidecarStatus);
  const startChatStream = useAppStore((s) => s.startChatStream);
  const appendChatToken = useAppStore((s) => s.appendChatToken);
  const appendChatEvent = useAppStore((s) => s.appendChatEvent);
  const endChatStream = useAppStore((s) => s.endChatStream);
  const setServiceStatus = useAppStore((s) => s.setServiceStatus);
  const pushToast = useAppStore((s) => s.pushToast);
  const dismissToast = useAppStore((s) => s.dismissToast);
  const toasts = useAppStore((s) => s.toasts);
  const hasCompletedFirstRun = useAppStore((s) => s.hasCompletedFirstRun);
  const setHasCompletedFirstRun = useAppStore((s) => s.setHasCompletedFirstRun);
  const setDockerStatus = useAppStore((s) => s.setDockerStatus);
  const startPowerModeRun = useAppStore((s) => s.startPowerModeRun);
  const upsertPowerModeStep = useAppStore((s) => s.upsertPowerModeStep);
  const addPowerModeApproval = useAppStore((s) => s.addPowerModeApproval);
  const setPowerModeMessage = useAppStore((s) => s.setPowerModeMessage);
  const setPowerModeError = useAppStore((s) => s.setPowerModeError);
  const endPowerModeRun = useAppStore((s) => s.endPowerModeRun);
  const pendingEscalations = useAppStore((s) => s.pendingEscalations);
  const setPendingEscalations = useAppStore((s) => s.setPendingEscalations);
  const addEscalation = useAppStore((s) => s.addEscalation);
  const removeEscalation = useAppStore((s) => s.removeEscalation);

  // ── Sidecar status subscription ────────────────────────────────────────
  useEffect(() => {
    let unsub: (() => void) | null = null;
    let alive = true;

    (async () => {
      // Pick up the current status snapshot synchronously so the StatusBar
      // doesn't flash "Initializing…" if the backend was already ready before
      // React mounted.
      const info = await window.electronAPI.getSidecarInfo();
      if (!alive) return;
      if (info) {
        setSidecarStatus({ status: "ready", port: info.port, token: info.token });
        resetSidecarInfo(info);
      }
      unsub = window.electronAPI.onSidecarStatus((status) => {
        setSidecarStatus(status);
        if (status.status === "ready") {
          resetSidecarInfo({ port: status.port, token: status.token });
        }
        if (status.status === "crashed" || status.status === "stopped") {
          resetSidecarInfo(null);
        }
      });
    })();

    return () => {
      alive = false;
      unsub?.();
    };
  }, [setSidecarStatus]);

  // ── SSE event stream wiring ────────────────────────────────────────────
  // Pull the primitives off the discriminated union so the effect's deps
  // are stable across status emits — without this, every setSidecarStatus
  // call (even with the same port/token) tears down and re-creates the
  // EventSource and drops in-flight events.
  const sidecarReady = sidecarStatus?.status === "ready";
  const sidecarPort = sidecarReady ? sidecarStatus.port : null;
  const sidecarToken = sidecarReady ? sidecarStatus.token : null;

  useEffect(() => {
    if (sidecarPort == null || sidecarToken == null) return;

    const sub = subscribeEvents(
      { port: sidecarPort, token: sidecarToken },
      {
        handlers: {
          chat_token: (data) => {
            const t = (data as { token?: string }).token ?? "";
            if (t) appendChatToken(t);
          },
          chat_event: (data) => {
            const evt = data as { type?: string };
            appendChatEvent(evt.type ?? "event", data);
          },
          chat_done: () => {
            endChatStream();
          },
          chat_stopped: () => {
            endChatStream();
          },
          chat_error: (data) => {
            const msg = (data as { error?: string }).error ?? "Chat failed";
            pushToast({ kind: "error", text: msg });
            endChatStream();
          },
          service_status_update: () => {
            // Refresh the whole snapshot when any service flips. Cheap.
            System.serviceStatus().then(setServiceStatus).catch(() => {});
          },
          service_unavailable: (data) => {
            const svc = (data as { service?: string }).service ?? "service";
            pushToast({
              kind: "warn",
              text: `${svc} is unavailable. Some features may be disabled.`,
            });
          },
          diagnostics_ready: (data) => {
            const path = (data as { path?: string }).path;
            if (path) {
              pushToast({ kind: "success", text: `Diagnostics saved to ${path}` });
            }
          },
          health_check_done: () => {
            pushToast({ kind: "info", text: "Health check complete" });
          },
          // ── Power Mode (v3) ──────────────────────────────────────────
          power_mode_status: (data) => {
            setDockerStatus(data as DockerStatusSnapshot);
          },
          power_mode_event: (data) => {
            const evt = data as { phase?: string; message?: string };
            if (evt.message && (evt.phase === "fatal" || evt.phase === "approval_timeout")) {
              pushToast({
                kind: evt.phase === "fatal" ? "error" : "warn",
                text: evt.message,
              });
            }
          },
          power_mode_started: (data) => {
            const evt = data as { task_id?: string; conversation_id?: string };
            if (evt.task_id && evt.conversation_id) {
              startPowerModeRun(evt.task_id, evt.conversation_id);
            }
          },
          power_mode_step: (data) => {
            const raw = data as Record<string, unknown>;
            const taskId = typeof raw.task_id === "string" ? raw.task_id : "";
            const stepId = typeof raw.step_id === "string" ? raw.step_id : "";
            if (!taskId || !stepId) return;
            // Spread the raw event first so any backend-provided fields are
            // captured, then write our normalized values last so they win
            // even when the backend omits a field (e.g. status defaults to
            // "done" rather than undefined).
            const step: ExecutionStep = {
              ...raw,
              step_id: stepId,
              kind: (typeof raw.kind === "string" ? raw.kind : "other") as ExecutionStepKind,
              status: ((raw.status as "running" | "done" | "error") ?? "done"),
            };
            upsertPowerModeStep(taskId, step);
          },
          power_mode_approval: (data) => {
            const evt = data as {
              task_id?: string;
              approval_id?: string;
              summary?: string;
              details?: Record<string, unknown>;
              danger?: "low" | "medium" | "high";
              timeout_sec?: number;
            };
            if (!evt.task_id || !evt.approval_id) return;
            addPowerModeApproval(evt.task_id, {
              approval_id: evt.approval_id,
              summary: evt.summary ?? "",
              details: evt.details ?? {},
              danger: evt.danger ?? "medium",
              expires_at: Date.now() + (evt.timeout_sec ?? 60) * 1000,
            });
          },
          power_mode_message: (data) => {
            const evt = data as { task_id?: string; text?: string };
            if (evt.task_id && evt.text) {
              setPowerModeMessage(evt.task_id, evt.text);
            }
          },
          power_mode_error: (data) => {
            const evt = data as { task_id?: string; error?: string };
            if (evt.task_id && evt.error) {
              setPowerModeError(evt.task_id, evt.error);
            } else if (evt.error) {
              pushToast({ kind: "error", text: evt.error });
            }
          },
          power_mode_done: (data) => {
            const evt = data as { task_id?: string };
            if (evt.task_id) endPowerModeRun(evt.task_id);
          },
          // ── Phase 5: Wiser-Human escalation channel ─────────────────
          escalation_required: (data) => {
            const evt = data as {
              escalation_id?: string;
              trigger_type?: string;
              trigger_detail?: string;
              conversation_id?: string;
            };
            if (!evt.escalation_id) return;
            const item: EscalationItem = {
              id: evt.escalation_id,
              conversation_id: evt.conversation_id ?? "",
              triggered_at: new Date().toISOString(),
              trigger_type: evt.trigger_type ?? "",
              trigger_detail: evt.trigger_detail ?? "",
            };
            addEscalation(item);
          },
          escalation_resolved: (data) => {
            const evt = data as { escalation_id?: string };
            if (evt.escalation_id) removeEscalation(evt.escalation_id);
          },
        },
        onError: (_err, { closed }) => {
          // EventSource handles transient blips on its own. Only act when
          // it has given up (readyState === CLOSED) — at that point ask
          // Electron for the current sidecar info; if the port changed
          // (e.g. user clicked Restart Backend) the new value will flow
          // through setSidecarStatus and this effect will re-subscribe
          // with the right URL.
          if (!closed) return;
          pushToast({
            kind: "warn",
            text: "Lost connection to backend. Reconnecting…",
          });
          window.electronAPI
            .getSidecarInfo()
            .then((info) => {
              if (info) {
                setSidecarStatus({
                  status: "ready",
                  port: info.port,
                  token: info.token,
                });
                resetSidecarInfo(info);
              }
            })
            .catch(() => {});
        },
      },
    );

    return () => {
      sub.close();
    };
  }, [
    sidecarPort,
    sidecarToken,
    appendChatToken,
    appendChatEvent,
    endChatStream,
    setServiceStatus,
    pushToast,
    setDockerStatus,
    startPowerModeRun,
    upsertPowerModeStep,
    addPowerModeApproval,
    setPowerModeMessage,
    setPowerModeError,
    endPowerModeRun,
    addEscalation,
    removeEscalation,
    setSidecarStatus,
  ]);

  // ── Pending escalations: hydrate on backend-ready ──────────────────────
  useEffect(() => {
    if (sidecarStatus?.status !== "ready") return;
    Escalation.pending()
      .then((rows) => {
        setPendingEscalations(rows as EscalationItem[]);
      })
      .catch(() => {});
  }, [sidecarStatus, setPendingEscalations]);

  // ── First-run check ────────────────────────────────────────────────────
  useEffect(() => {
    if (sidecarStatus?.status !== "ready") return;
    Settings.get()
      .then((s) => {
        setHasCompletedFirstRun(!!s.first_run_complete);
      })
      .catch(() => {});
  }, [sidecarStatus, setHasCompletedFirstRun]);

  // The first pending escalation drives the modal. New ones queue in
  // pendingEscalations; once the user resolves the head, the next becomes
  // the active modal on the next render.
  const activeEscalation =
    hasCompletedFirstRun && sidecarStatus?.status === "ready"
      ? pendingEscalations[0] ?? null
      : null;

  return (
    <div className="flex flex-col h-screen">
      <StatusBar />
      <div className="flex flex-1 min-h-0">
        <Sidebar />
        <main className="flex-1 min-w-0 overflow-hidden">
          {view === "chat" && <ChatView />}
          {view === "agents" && <AgentPanel />}
          {view === "rag" && <RagPanel />}
          {view === "memory" && <MemoryPanel />}
          {view === "prompts" && <PromptPanel />}
          {view === "mcp" && <McpPanel />}
          {view === "security" && <SecurityPanel />}
          {view === "settings" && <SettingsPanel />}
          {view === "diagnostics" && <DiagnosticsPanel />}
          {view === "escalations" && <EscalationPanel />}
        </main>
      </div>

      {!hasCompletedFirstRun && sidecarStatus?.status === "ready" && (
        <FirstRunWizard onComplete={() => setHasCompletedFirstRun(true)} />
      )}

      {activeEscalation && <EscalationModal escalation={activeEscalation} />}

      <div className="fixed bottom-4 right-4 flex flex-col gap-2 z-40">
        {toasts.map((t) => {
          const tone =
            t.kind === "error"
              ? "border-err/40 text-err bg-err/10"
              : t.kind === "warn"
                ? "border-warn/40 text-warn bg-warn/10"
                : t.kind === "success"
                  ? "border-ok/40 text-ok bg-ok/10"
                  : "border-line text-ink bg-bg-2";
          return (
            <button
              key={t.id}
              onClick={() => dismissToast(t.id)}
              className={`max-w-sm text-left text-sm px-3 py-2 rounded-md border ${tone}`}
            >
              {t.text}
            </button>
          );
        })}
      </div>
    </div>
  );
}
