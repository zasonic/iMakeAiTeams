import { useEffect, useState } from "react";

import { Escalation as EscalationApi } from "@/api/client";
import { useAppStore, type Escalation } from "@/stores/appStore";

const TRIGGER_LABEL: Record<string, string> = {
  replacement_threat: "Replacement threat",
  autonomy_reduction: "Autonomy reduction",
  goal_conflict: "Goal conflict",
};

export function EscalationPanel() {
  const ready = useAppStore((s) => s.sidecarStatus?.status === "ready");
  const pendingEscalations = useAppStore((s) => s.pendingEscalations);
  const setPendingEscalations = useAppStore((s) => s.setPendingEscalations);
  const removeEscalation = useAppStore((s) => s.removeEscalation);
  const pushToast = useAppStore((s) => s.pushToast);
  const [busyId, setBusyId] = useState<string | null>(null);

  useEffect(() => {
    if (!ready) return;
    EscalationApi.pending()
      .then((rows) => setPendingEscalations(rows as Escalation[]))
      .catch(() => {});
  }, [ready, setPendingEscalations]);

  const resolve = async (id: string, decision: "approve" | "deny") => {
    setBusyId(id);
    try {
      const fn = decision === "approve" ? EscalationApi.approve : EscalationApi.deny;
      const rsp = await fn(id);
      if (rsp.ok) {
        removeEscalation(id);
      } else {
        pushToast({
          kind: "error",
          text: rsp.error ?? "Could not resolve escalation",
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
    <div className="p-6 overflow-y-auto h-full max-w-3xl">
      <header className="mb-4">
        <h1 className="text-xl font-semibold">Pending reviews</h1>
        <p className="text-sm text-ink-dim">
          Actions paused for your approval because of a Wiser-Human
          escalation trigger. Approve to let the model continue, or Deny to
          block the action.
        </p>
      </header>

      <ul className="space-y-2">
        {pendingEscalations.map((e) => {
          const triggerLabel = TRIGGER_LABEL[e.trigger_type] ?? e.trigger_type;
          const busy = busyId === e.id;
          return (
            <li key={e.id} className="card">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-xs uppercase tracking-wide text-warn">
                    {triggerLabel}
                  </div>
                  <div className="mt-1 text-sm font-mono break-words">
                    {e.trigger_detail || "(no detail)"}
                  </div>
                  {e.model_input ? (
                    <div className="mt-2 text-xs text-ink-dim whitespace-pre-wrap max-h-24 overflow-y-auto">
                      {e.model_input}
                    </div>
                  ) : null}
                  <div className="text-[11px] text-ink-faint mt-2">
                    {e.triggered_at?.slice(0, 19).replace("T", " ")} ·{" "}
                    {e.conversation_id.slice(0, 8)}
                  </div>
                </div>
                <div className="flex flex-col gap-2 shrink-0">
                  <button
                    type="button"
                    className="btn-primary"
                    onClick={() => resolve(e.id, "approve")}
                    disabled={busy}
                  >
                    Approve
                  </button>
                  <button
                    type="button"
                    className="btn-danger"
                    onClick={() => resolve(e.id, "deny")}
                    disabled={busy}
                  >
                    Deny
                  </button>
                </div>
              </div>
            </li>
          );
        })}
        {!pendingEscalations.length && (
          <li className="text-ink-faint text-sm">No pending reviews.</li>
        )}
      </ul>
    </div>
  );
}
