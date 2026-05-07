import { useEffect, useState } from "react";

import { Memory } from "@/api/client";
import { useAppStore, type PendingWrite } from "@/stores/appStore";

export function MemoryReviewPanel() {
  const ready = useAppStore((s) => s.sidecarStatus?.status === "ready");
  const pendingMemoryWrites = useAppStore((s) => s.pendingMemoryWrites);
  const setPendingMemoryWrites = useAppStore((s) => s.setPendingMemoryWrites);
  const removePendingMemoryWrite = useAppStore((s) => s.removePendingMemoryWrite);
  const pushToast = useAppStore((s) => s.pushToast);
  const [busyId, setBusyId] = useState<string | null>(null);

  useEffect(() => {
    if (!ready) return;
    Memory.pending()
      .then((rows) => setPendingMemoryWrites(rows as PendingWrite[]))
      .catch(() => {});
  }, [ready, setPendingMemoryWrites]);

  const resolve = async (id: string, decision: "approve" | "deny") => {
    setBusyId(id);
    try {
      const fn = decision === "approve" ? Memory.approvePending : Memory.denyPending;
      const rsp = await fn(id);
      if (rsp.ok) {
        removePendingMemoryWrite(id);
      } else {
        pushToast({
          kind: "error",
          text: rsp.error ?? "Could not resolve memory write",
        });
      }
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Network error",
      });
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div className="p-6 overflow-y-auto h-full max-w-4xl">
      <header className="mb-4">
        <h1 className="text-xl font-semibold">Memory review</h1>
        <p className="text-sm text-ink-dim">
          Auto-extracted facts that contradict existing memory are paused
          here. Approve to overwrite the existing memory with the new fact,
          or Deny to discard the new fact and keep what's stored.
        </p>
      </header>

      <ul className="space-y-2">
        {pendingMemoryWrites.map((w) => {
          const busy = busyId === w.id;
          return (
            <li key={w.id} className="card">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <div className="text-xs uppercase tracking-wide text-warn">
                    {w.write_type} · contradicts existing memory
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-2">
                    <div className="border border-line rounded-md p-2 bg-bg-2/40">
                      <div className="text-[11px] uppercase text-ink-faint mb-1">
                        New fact (proposed)
                      </div>
                      <div className="text-sm whitespace-pre-wrap break-words">
                        {w.content}
                      </div>
                    </div>
                    <div className="border border-line rounded-md p-2 bg-bg-2/40">
                      <div className="text-[11px] uppercase text-ink-faint mb-1">
                        Existing fact (contradicted)
                      </div>
                      <div className="text-sm whitespace-pre-wrap break-words">
                        {w.contradicts_content ?? "(no matching fact found)"}
                      </div>
                    </div>
                  </div>
                  <div className="text-[11px] text-ink-faint mt-2">
                    {w.proposed_at?.slice(0, 19).replace("T", " ")}
                    {w.conversation_id ? ` · ${w.conversation_id.slice(0, 8)}` : ""}
                  </div>
                </div>
                <div className="flex flex-col gap-2 shrink-0">
                  <button
                    type="button"
                    className="btn-primary"
                    onClick={() => resolve(w.id, "approve")}
                    disabled={busy}
                  >
                    Approve
                  </button>
                  <button
                    type="button"
                    className="btn-danger"
                    onClick={() => resolve(w.id, "deny")}
                    disabled={busy}
                  >
                    Deny
                  </button>
                </div>
              </div>
            </li>
          );
        })}
        {!pendingMemoryWrites.length && (
          <li className="text-ink-faint text-sm">No memory writes awaiting review.</li>
        )}
      </ul>
    </div>
  );
}
