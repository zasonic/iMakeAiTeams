// src/components/ChatView.tsx — chat panel with conversation list + streaming.

import { useEffect, useMemo, useRef, useState } from "react";

import { Chat } from "@/api/client";
import { useAppStore } from "@/stores/appStore";

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

  const [conversations, setConversations] = useState<ConversationRow[]>([]);
  const [activeId, setActiveId] = useState<string>("");
  const [messages, setMessages] = useState<MessageRow[]>([]);
  const [input, setInput] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [loadError, setLoadError] = useState<string>("");

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const ready = status?.status === "ready";

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

  // Auto-scroll on new tokens.
  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, activeChat?.buffer]);

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
    const text = input;
    setInput("");
    setBusy(true);
    startChatStream(activeId);
    setMessages((prev) => [
      ...prev,
      {
        id: `local-${Date.now()}`,
        role: "user",
        content: text,
      },
    ]);
    try {
      await Chat.send(activeId, text);
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "chat send failed",
      });
      setBusy(false);
      endChatStream();
    }
  };

  // When the SSE handler in App.tsx fires endChatStream, also drop the busy
  // flag and reload messages so the persisted assistant message replaces the
  // streaming buffer.
  useEffect(() => {
    if (!busy) return;
    if (activeChat) return; // still streaming
    setBusy(false);
    if (!activeId) return;
    Chat.messages(activeId)
      .then((rows) => setMessages(rows as MessageRow[]))
      .catch(() => {});
  }, [activeChat, busy, activeId]);

  const streamingBuffer = activeChat?.conversationId === activeId ? activeChat.buffer : "";

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
              placeholder={ready ? "Type a message…" : "Waiting for backend…"}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
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
              <button
                className="btn-ghost"
                onClick={() => Chat.stop().catch(() => {})}
              >
                Stop
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
