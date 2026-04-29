import { useEffect, useState } from "react";

import { Memory } from "@/api/client";
import { useAppStore } from "@/stores/appStore";

interface MemoryRow {
  id: string;
  content: string;
  category?: string;
  source?: string;
  created_at?: string;
}

export function MemoryPanel() {
  const ready = useAppStore((s) => s.sidecarStatus?.status === "ready");
  const pushToast = useAppStore((s) => s.pushToast);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<MemoryRow[]>([]);
  const [draft, setDraft] = useState("");

  const search = async () => {
    if (!query.trim()) return;
    try {
      const rows = await Memory.searchMemories(query);
      setResults(rows as MemoryRow[]);
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Search failed",
      });
    }
  };

  const save = async () => {
    if (!draft.trim()) return;
    try {
      await Memory.save(draft);
      pushToast({ kind: "success", text: "Memory saved" });
      setDraft("");
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Save failed",
      });
    }
  };

  return (
    <div className="p-6 overflow-y-auto h-full max-w-3xl">
      <header className="mb-4">
        <h1 className="text-xl font-semibold">Memory</h1>
        <p className="text-sm text-ink-dim">
          Long-term facts surfaced as system context for chats and agents.
        </p>
      </header>

      <div className="card mb-4">
        <h3 className="font-semibold mb-2">Save a fact</h3>
        <textarea
          className="input min-h-[80px]"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="e.g. Customer preference for concise answers"
          disabled={!ready}
        />
        <div className="mt-2">
          <button className="btn-primary" onClick={save} disabled={!ready || !draft.trim()}>
            Save
          </button>
        </div>
      </div>

      <div className="card">
        <h3 className="font-semibold mb-2">Search</h3>
        <div className="flex gap-2">
          <input
            className="input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && search()}
            placeholder="Semantic search across stored facts…"
            disabled={!ready}
          />
          <button className="btn-primary" onClick={search} disabled={!ready}>
            Search
          </button>
        </div>
        <ul className="mt-3 space-y-2">
          {results.map((r) => (
            <li key={r.id} className="border border-line rounded-md p-2 bg-bg-2/40">
              <div className="text-sm">{r.content}</div>
              <div className="text-[11px] text-ink-faint mt-1">
                {r.category} · {r.source} · {r.created_at?.slice(0, 16)}
              </div>
            </li>
          ))}
          {!results.length && (
            <li className="text-ink-faint text-sm">No matches.</li>
          )}
        </ul>
      </div>
    </div>
  );
}
