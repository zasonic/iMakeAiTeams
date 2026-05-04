import { useEffect, useState } from "react";

import { Memory, Rag } from "@/api/client";
import { useAppStore } from "@/stores/appStore";

interface RagStatus {
  total_chunks?: number;
  total_size?: number;
  last_indexed?: string;
}

export function RagPanel() {
  const ready = useAppStore((s) => s.sidecarStatus?.status === "ready");
  const pushToast = useAppStore((s) => s.pushToast);
  const [status, setStatus] = useState<RagStatus>({});
  const [available, setAvailable] = useState<boolean>(false);
  const [busy, setBusy] = useState(false);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<unknown[]>([]);

  useEffect(() => {
    if (!ready) return;
    Rag.status().then((s) => setStatus(s as RagStatus)).catch(() => {});
    Memory.semanticAvailable()
      .then(({ available }) => setAvailable(available))
      .catch(() => setAvailable(false));
  }, [ready]);

  const indexFolder = async () => {
    const folder = await window.electronAPI.selectFolder();
    if (!folder) return;
    setBusy(true);
    try {
      await Rag.indexFolder(folder);
      pushToast({ kind: "success", text: `Indexed ${folder}` });
      setStatus(await Rag.status() as RagStatus);
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Index failed",
      });
    } finally {
      setBusy(false);
    }
  };

  const search = async () => {
    if (!query.trim()) return;
    try {
      const rows = await Rag.searchHybrid(query);
      setResults(rows);
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Search failed",
      });
    }
  };

  return (
    <div className="p-6 overflow-y-auto h-full">
      <header className="mb-4">
        <h1 className="text-xl font-semibold">Documents (RAG)</h1>
        <p className="text-sm text-ink-dim">
          Index folders for hybrid (BM25 + semantic) retrieval. Semantic search
          requires the optional ML stack — install via the “full” bundle.
        </p>
      </header>

      {!available && (
        <div className="card mb-4 border-warn/30 text-warn text-sm">
          Semantic search is unavailable in this build. BM25 keyword search
          still works for indexed documents.
        </div>
      )}

      <div className="card mb-4">
        <h3 className="font-semibold mb-2">Library</h3>
        <div className="text-sm text-ink-dim space-y-1">
          <div>Chunks: {status.total_chunks ?? 0}</div>
          {status.last_indexed && <div>Last indexed: {status.last_indexed}</div>}
        </div>
        <div className="mt-3 flex gap-2">
          <button className="btn-primary" onClick={indexFolder} disabled={!ready || busy}>
            {busy ? "Indexing…" : "Index folder"}
          </button>
          <button
            className="btn-ghost"
            onClick={async () => {
              try {
                await Rag.clear();
                setStatus(await Rag.status() as RagStatus);
                pushToast({ kind: "success", text: "RAG index cleared" });
              } catch (err) {
                pushToast({
                  kind: "error",
                  text: err instanceof Error ? err.message : "Clear failed",
                });
              }
            }}
            disabled={!ready || busy}
          >
            Clear
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
            placeholder="Search indexed documents…"
            disabled={!ready}
          />
          <button className="btn-primary" onClick={search} disabled={!ready}>
            Search
          </button>
        </div>
        {results.length > 0 && (
          <ul className="mt-3 space-y-2 text-sm">
            {results.map((r, i) => (
              <li key={i} className="border border-line rounded-md p-2 bg-bg-2/40">
                <pre className="whitespace-pre-wrap break-words font-mono text-xs">
                  {JSON.stringify(r, null, 2)}
                </pre>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
