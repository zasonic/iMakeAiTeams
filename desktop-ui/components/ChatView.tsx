// desktop-ui/components/ChatView.tsx — chat panel with conversation list + streaming.
//
// v3 adds Power Mode: when `power_mode_enabled`, the renderer asks the
// backend to classify each message; "execution"-class messages go to
// /api/docker/execute and stream OpenClaw step events via SSE. "Chat"-class
// messages keep the v2 path intact.

import { useEffect, useMemo, useRef, useState } from "react";

import { Chat, Docker, Settings } from "@/api/client";
import { ExecutionCard } from "@/components/ExecutionCard";
import { useAppStore, type PowerModeRun } from "@/stores/appStore";

interface ConversationRow {
  id: string;
  title?: string;
  updated_at?: string;
}

interface MessageRow {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  model_used?: string;
  cost_usd?: number;
}

export function ChatView() {
  const status = useAppStore((s) => s.sidecarStatus);
  const activeChat = useAppStore((s) => s.activeChat);
  const startChatStream = useAppStore((s) => s.startChatStream);
  const endChatStream = useAppStore((s) => s.endChatStream);
  const pushToast = useAppStore((s) => s.pushToast);
  const powerModeRuns = useAppStore((s) => s.powerModeRuns);
  const resolvePowerModeApproval = useAppStore((s) => s.resolvePowerModeApproval);
  const powerModeEnabled = useAppStore((s) => s.powerModeEnabled);
  const setPowerModeEnabled = useAppStore((s) => s.setPowerModeEnabled);

  const [conversations, setConversations] = useState<ConversationRow[]>([]);
  const [activeId, setActiveId] = useState<string>("");
  const [messages, setMessages] = useState<MessageRow[]>([]);
  const [input, setInput] = useState<string>("");
  // Send phase explicitly drives the Send button's disabled state and the
  // cleanup effects below. "classifying" covers the (potentially LLM-backed)
  // classify round-trip; while in that state, neither the chat-stream nor
  // Power Mode cleanup effects should fire.
  const [sendPhase, setSendPhase] = useState<
    "idle" | "classifying" | "chat" | "execution"
  >("idle");
  const [activeTaskId, setActiveTaskId] = useState<string>("");
  const [loadError, setLoadError] = useState<string>("");

  const busy = sendPhase !== "idle";

  const scrollRef = useRef<HTMLDivElement | null>(null);
  // Synchronous lock so two near-simultaneous Enter presses can't both pass
  // the `busy` guard before React re-renders with sendPhase="classifying".
  const sendLockRef = useRef(false);
  // Tracks whether the user is mid-IME composition. CJK input methods fire
  // Enter to commit a composition, which would otherwise submit the form
  // and lose the half-typed glyph.
  const composingRef = useRef(false);
  const ready = status?.status === "ready";

  // Sync the Power Mode flag from the sidecar on first ready. After that the
  // appStore owns the value — SettingsPanel updates it whenever the toggle
  // changes, so a refetch here would race with that.
  useEffect(() => {
    if (!ready) return;
    let alive = true;
    Settings.get()
      .then((s) => {
        if (alive) setPowerModeEnabled(!!s.power_mode_enabled);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [ready, setPowerModeEnabled]);

  // Load conversation list once the sidecar is ready.
  useEffect(() => {
    if (!ready) return;
    let alive = true;
    (async () => {
      try {
        const rows = (await Chat.list(50)) as ConversationRow[];
        if (alive) setConversations(rows);
        if (alive && rows.length && !activeId) setActiveId(rows[0].id);
      } catch (err) {
        if (alive) setLoadError(err instanceof Error ? err.message : String(err));
      }
    })();
    return () => {
      alive = false;
    };
  }, [ready, activeId]);

  // Load messages when active conversation changes.
  useEffect(() => {
    if (!ready || !activeId) return;
    let alive = true;
    (async () => {
      try {
        const rows = (await Chat.messages(activeId)) as MessageRow[];
        if (alive) setMessages(rows);
      } catch (err) {
        if (alive) setLoadError(err instanceof Error ? err.message : String(err));
      }
    })();
    return () => {
      alive = false;
    };
  }, [ready, activeId]);

  // Auto-scroll on new tokens / steps. Use "auto" (instant) while a stream
  // is active so per-token scrolls don't queue smooth animations that stutter
  // on long responses; use "smooth" for the rare list-level changes.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const streaming = !!activeChat?.buffer;
    el.scrollTo({
      top: el.scrollHeight,
      behavior: streaming ? "auto" : "smooth",
    });
  }, [messages, activeChat?.buffer, powerModeRuns]);

  const newConversation = async () => {
    try {
      const { id } = await Chat.newConversation();
      setActiveId(id);
      const rows = (await Chat.list(50)) as ConversationRow[];
      setConversations(rows);
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Failed to create conversation",
      });
    }
  };

  const send = async () => {
    if (!activeId || !input.trim() || busy) return;
    if (sendLockRef.current) return;
    sendLockRef.current = true;
    const text = input;
    setInput("");
    setSendPhase("classifying");
    setMessages((prev) => [
      ...prev,
      { id: `local-${Date.now()}`, role: "user", content: text },
    ]);

    let routedToExecution = false;
    if (powerModeEnabled) {
      try {
        const verdict = await Docker.classify(text, activeId);
        if (verdict.route === "execution") {
          const health = await Docker.health().catch(() => ({ ok: false } as { ok: boolean }));
          if (!health.ok) {
            // OpenClaw isn't healthy — fall back to chat with a hint.
            pushToast({
              kind: "warn",
              text: "Power Mode is enabled but OpenClaw isn't running. Falling back to chat.",
            });
          } else {
            const r = await Docker.execute(activeId, text);
            if (r.ok && r.task_id) {
              setActiveTaskId(r.task_id);
              setSendPhase("execution");
              routedToExecution = true;
            } else if (r.error) {
              pushToast({ kind: "error", text: r.error });
            }
          }
        }
      } catch (err) {
        // Classifier failure shouldn't block the user — fall back to chat.
        console.warn("classify failed:", err);
      }
    } else {
      // Power Mode off — but if the message *looks* execution-class, hint at it.
      try {
        const verdict = await Docker.classify(text, activeId);
        if (verdict.route === "execution") {
          pushToast({
            kind: "info",
            text:
              "I can do this for you if you enable Power Mode in Settings.",
          });
        }
      } catch {
        /* classifier is optional in this path */
      }
    }

    if (routedToExecution) {
      sendLockRef.current = false;
      return;
    }

    setSendPhase("chat");
    startChatStream(activeId);
    try {
      await Chat.send(activeId, text);
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "chat send failed",
      });
      setSendPhase("idle");
      endChatStream();
    } finally {
      sendLockRef.current = false;
    }
  };

  // When a chat stream ends, drop busy and reload persisted messages.
  // Only fires for the chat path; Power Mode and classifying phases use their
  // own cleanup so this effect can't clear busy mid-flight.
  useEffect(() => {
    if (sendPhase !== "chat") return;
    if (activeChat) return; // chat still streaming
    setSendPhase("idle");
    if (!activeId) return;
    Chat.messages(activeId)
      .then((rows) => setMessages(rows as MessageRow[]))
      .catch(() => {});
  }, [activeChat, sendPhase, activeId]);

  // Watchdog: if the sidecar dies (or a chat_done event is lost) while we're
  // in the chat phase, reset the Send button instead of leaving it stuck.
  useEffect(() => {
    if (sendPhase !== "chat") return;
    if (ready) return;
    setSendPhase("idle");
    endChatStream();
  }, [ready, sendPhase, endChatStream]);

  // When the active Power Mode run finishes, drop busy and clear the task id
  // so a new send doesn't think the old (already-done) task is still active.
  const activeRun = activeTaskId ? powerModeRuns[activeTaskId] : null;
  useEffect(() => {
    if (sendPhase !== "execution") return;
    if (!activeTaskId) return;
    if (!activeRun) return;
    if (activeRun.done) {
      setSendPhase("idle");
      setActiveTaskId("");
    }
  }, [activeRun, activeTaskId, sendPhase]);

  const cancelActive = async () => {
    if (sendPhase === "execution" && activeTaskId) {
      try {
        await Docker.cancel(activeTaskId);
      } catch {
        /* ignore */
      }
      setActiveTaskId("");
      setSendPhase("idle");
      return;
    }
    if (sendPhase === "chat") {
      Chat.stop().catch(() => {});
      // The chat stream end effect will flip back to "idle".
      return;
    }
    // Classifying — abandon the in-flight request and reset.
    setSendPhase("idle");
  };

  const approve = async (taskId: string, approvalId: string, allow: boolean) => {
    try {
      await Docker.approve(taskId, approvalId, allow);
      resolvePowerModeApproval(taskId, approvalId);
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Approval failed",
      });
    }
  };

  const streamingBuffer = activeChat?.conversationId === activeId ? activeChat.buffer : "";

  // Power Mode runs scoped to the active conversation, in order.
  const conversationRuns = useMemo(() => {
    return Object.values(powerModeRuns)
      .filter((r) => r.conversationId === activeId)
      .sort((a, b) => a.startedAt - b.startedAt);
  }, [powerModeRuns, activeId]);

  return (
    <div className="flex h-full">
      <div className="w-64 border-r border-line bg-bg-1 flex flex-col">
        <div className="p-3 border-b border-line">
          <button className="btn-primary w-full" onClick={newConversation}>
            + New conversation
          </button>
        </div>
        <div className="flex-1 overflow-y-auto">
          {conversations.map((c) => (
            <button
              key={c.id}
              type="button"
              onClick={() => setActiveId(c.id)}
              className={`w-full text-left px-4 py-2 text-sm border-b border-line/30 ${
                c.id === activeId
                  ? "bg-accent/10 text-ink"
                  : "text-ink-dim hover:bg-bg-2"
              }`}
            >
              <div className="truncate font-medium">{c.title || "Untitled"}</div>
              <div className="text-[11px] text-ink-faint">{c.updated_at?.slice(0, 16)}</div>
            </button>
          ))}
          {!conversations.length && !loadError && (
            <div className="p-4 text-sm text-ink-faint">No conversations yet.</div>
          )}
          {loadError && (
            <div className="p-4 text-sm text-err">{loadError}</div>
          )}
        </div>
      </div>

      <div className="flex-1 flex flex-col min-w-0">
        <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-4 space-y-3">
          {messages.map((m) => (
            <div
              key={m.id}
              className={`max-w-[80%] rounded-xl px-4 py-2 text-sm whitespace-pre-wrap ${
                m.role === "user"
                  ? "ml-auto bg-accent/15 text-ink border border-accent/20"
                  : "bg-bg-2 text-ink border border-line"
              }`}
            >
              {m.content}
              {m.model_used && (
                <div className="text-[11px] text-ink-faint mt-2">
                  {m.model_used}
                  {typeof m.cost_usd === "number" && ` · $${m.cost_usd.toFixed(4)}`}
                </div>
              )}
            </div>
          ))}

          {conversationRuns.map((run) => (
            <PowerModeMessage
              key={run.taskId}
              run={run}
              onApprove={(approvalId, allow) => approve(run.taskId, approvalId, allow)}
              onCancel={() => Docker.cancel(run.taskId).catch(() => {})}
            />
          ))}

          {streamingBuffer && (
            <div className="max-w-[80%] rounded-xl px-4 py-2 text-sm whitespace-pre-wrap bg-bg-2 text-ink border border-line">
              {streamingBuffer}
              <span className="inline-block ml-1 h-3 w-1 bg-accent animate-pulse align-middle" />
            </div>
          )}
        </div>

        <div className="border-t border-line p-3">
          <div className="flex gap-2 items-end">
            <textarea
              className="input flex-1 min-h-[44px] max-h-40 resize-none"
              placeholder={
                ready
                  ? powerModeEnabled
                    ? "Type a message… (Power Mode is on)"
                    : "Type a message…"
                  : "Waiting for backend…"
              }
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onCompositionStart={() => {
                composingRef.current = true;
              }}
              onCompositionEnd={() => {
                composingRef.current = false;
              }}
              onKeyDown={(e) => {
                if (e.key !== "Enter" || e.shiftKey) return;
                // Don't submit while an IME composition is in flight (e.g.
                // Japanese / Chinese / Korean input). isComposing covers both
                // the keydown that commits a composition (which fires after
                // compositionend on some browsers) and key 229 events.
                if (composingRef.current || e.nativeEvent.isComposing) return;
                e.preventDefault();
                send();
              }}
              disabled={!ready || !activeId || busy}
            />
            <button
              className="btn-primary"
              onClick={send}
              disabled={!ready || !activeId || busy || !input.trim()}
            >
              Send
            </button>
            {busy && (
              <button className="btn-ghost" onClick={cancelActive}>
                Stop
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Power Mode message bubble ───────────────────────────────────────────────

interface PowerModeMessageProps {
  run: PowerModeRun;
  onApprove: (approvalId: string, allow: boolean) => void;
  onCancel: () => void;
}

function PowerModeMessage({ run, onApprove, onCancel }: PowerModeMessageProps) {
  const elapsed = Math.max(0, Math.floor((Date.now() - run.startedAt) / 1000));
  const showProgress = !run.done;

  return (
    <div className="max-w-[85%] rounded-xl px-4 py-3 text-sm bg-bg-2 text-ink border border-line space-y-2">
      <div className="flex items-center gap-2 text-[11px] text-ink-faint">
        <span className="px-1.5 py-0.5 rounded bg-accent/15 text-accent border border-accent/30 font-semibold">
          ⚡ Power Mode
        </span>
        <span className="font-mono">{run.taskId}</span>
        {showProgress && (
          <span className="text-warn">working… {elapsed}s</span>
        )}
        {showProgress && (
          <button type="button" className="ml-auto text-ink-dim hover:text-ink" onClick={onCancel}>
            Cancel
          </button>
        )}
      </div>

      {run.steps.length > 0 && (
        <div className="space-y-1.5">
          {run.steps.map((step) => (
            <ExecutionCard key={step.step_id} step={step} />
          ))}
        </div>
      )}

      {run.approvals.map((appr) => (
        <ApprovalCard
          key={appr.approval_id}
          summary={appr.summary}
          details={appr.details}
          danger={appr.danger}
          expiresAt={appr.expires_at}
          onAllow={() => onApprove(appr.approval_id, true)}
          onDeny={() => onApprove(appr.approval_id, false)}
        />
      ))}

      {run.resultText && (
        <div className="whitespace-pre-wrap text-sm">{run.resultText}</div>
      )}

      {run.error && (
        <div className="rounded-md border border-err/40 bg-err/5 px-3 py-2 text-err text-xs">
          {run.error}
        </div>
      )}

      {run.done && !run.error && !run.resultText && (
        <div className="text-[11px] text-ink-faint">Cancelled by user</div>
      )}
    </div>
  );
}

interface ApprovalCardProps {
  summary: string;
  details: Record<string, unknown>;
  danger: "low" | "medium" | "high";
  expiresAt: number;
  onAllow: () => void;
  onDeny: () => void;
}

function ApprovalCard({
  summary,
  details,
  danger,
  expiresAt,
  onAllow,
  onDeny,
}: ApprovalCardProps) {
  const [, force] = useState(0);
  const remaining = Math.max(0, Math.ceil((expiresAt - Date.now()) / 1000));
  useEffect(() => {
    // Run a single 1s timer per approval and stop ticking once the deadline
    // has passed. The effect re-runs only when expiresAt changes, so we
    // don't accumulate timers across re-renders.
    if (Date.now() >= expiresAt) return;
    const id = window.setInterval(() => {
      force((n) => n + 1);
      if (Date.now() >= expiresAt) window.clearInterval(id);
    }, 1000);
    return () => window.clearInterval(id);
  }, [expiresAt]);
  const tone = danger === "high"
    ? "border-err/50 bg-err/5"
    : danger === "low"
      ? "border-line bg-bg-1"
      : "border-warn/40 bg-warn/5";

  return (
    <div className={`rounded-md border ${tone} px-3 py-2 text-sm space-y-2`}>
      <div className="flex items-center justify-between">
        <span className="font-semibold">Approval needed</span>
        <span className="text-[11px] text-ink-faint">
          auto-deny in {remaining}s
        </span>
      </div>
      <p className="text-ink">{summary || "OpenClaw wants to perform an action."}</p>
      {Object.keys(details).length > 0 && (
        <pre className="text-[11px] font-mono whitespace-pre-wrap text-ink-dim border-t border-line/60 pt-2">
          {JSON.stringify(details, null, 2)}
        </pre>
      )}
      <div className="flex gap-2 pt-1">
        <button type="button" className="btn-primary text-xs" onClick={onAllow}>
          Allow
        </button>
        <button type="button" className="btn-ghost text-xs" onClick={onDeny}>
          Deny
        </button>
      </div>
    </div>
  );
}
