import { useEffect } from "react";

import { Settings, System, resetSidecarInfo } from "@/api/client";
import { subscribeEvents, closeEventStream } from "@/api/sse";
import { AgentPanel } from "@/components/AgentPanel";
import { ChatView } from "@/components/ChatView";
import { DiagnosticsPanel } from "@/components/DiagnosticsPanel";
import { FirstRunWizard } from "@/components/FirstRunWizard";
import { McpPanel } from "@/components/McpPanel";
import { MemoryPanel } from "@/components/MemoryPanel";
import { PromptPanel } from "@/components/PromptPanel";
import { RagPanel } from "@/components/RagPanel";
import { SecurityPanel } from "@/components/SecurityPanel";
import { SettingsPanel } from "@/components/SettingsPanel";
import { Sidebar } from "@/components/Sidebar";
import { StatusBar } from "@/components/StatusBar";
import { useAppStore } from "@/stores/appStore";

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
          closeEventStream();
        }
      });
    })();

    return () => {
      alive = false;
      unsub?.();
      closeEventStream();
    };
  }, [setSidecarStatus]);

  // ── SSE event stream wiring ────────────────────────────────────────────
  useEffect(() => {
    if (sidecarStatus?.status !== "ready") return;

    const sub = subscribeEvents(
      { port: sidecarStatus.port, token: sidecarStatus.token },
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
        },
        onError: () => {
          // EventSource auto-reconnects; only surface persistent failures.
        },
      },
    );

    return () => {
      sub.close();
    };
  }, [
    sidecarStatus,
    appendChatToken,
    appendChatEvent,
    endChatStream,
    setServiceStatus,
    pushToast,
  ]);

  // ── First-run check ────────────────────────────────────────────────────
  useEffect(() => {
    if (sidecarStatus?.status !== "ready") return;
    Settings.get()
      .then((s) => {
        setHasCompletedFirstRun(!!s.first_run_complete);
      })
      .catch(() => {});
  }, [sidecarStatus, setHasCompletedFirstRun]);

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
        </main>
      </div>

      {!hasCompletedFirstRun && sidecarStatus?.status === "ready" && (
        <FirstRunWizard onComplete={() => setHasCompletedFirstRun(true)} />
      )}

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
