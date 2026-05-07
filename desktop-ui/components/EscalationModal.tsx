import { useState } from "react";

import { Escalation as EscalationApi } from "@/api/client";
import { useAppStore, type Escalation } from "@/stores/appStore";

const TRIGGER_LABEL: Record<string, string> = {
  replacement_threat: "Replacement threat",
  autonomy_reduction: "Autonomy reduction",
  goal_conflict: "Goal conflict",
};

interface Props {
  escalation: Escalation;
}

export function EscalationModal({ escalation }: Props) {
  const removeEscalation = useAppStore((s) => s.removeEscalation);
  const pushToast = useAppStore((s) => s.pushToast);
  const [busy, setBusy] = useState(false);

  const resolve = async (decision: "approve" | "deny") => {
    if (busy) return;
    setBusy(true);
    try {
      const fn = decision === "approve" ? EscalationApi.approve : EscalationApi.deny;
      const rsp = await fn(escalation.id);
      if (rsp.ok) {
        removeEscalation(escalation.id);
        pushToast({
          kind: decision === "approve" ? "success" : "info",
          text: `Escalation ${decision === "approve" ? "approved" : "denied"}`,
        });
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
      setBusy(false);
    }
  };

  const triggerLabel = TRIGGER_LABEL[escalation.trigger_type] ?? escalation.trigger_type;

  return (
    <div
      className="fixed inset-0 z-50 bg-bg/95 flex items-center justify-center p-6"
      role="dialog"
      aria-modal="true"
      aria-labelledby="escalation-modal-title"
    >
      <div className="card max-w-lg w-full">
        <header className="mb-4">
          <div className="text-xs uppercase tracking-wide text-warn mb-1">
            Human review required
          </div>
          <h1 id="escalation-modal-title" className="text-xl font-semibold">
            {triggerLabel}
          </h1>
        </header>

        <div className="space-y-3 text-sm">
          <div>
            <div className="label">Trigger detail</div>
            <div className="rounded-md border border-line bg-bg-2/40 px-3 py-2 text-ink font-mono text-xs">
              {escalation.trigger_detail || "(no detail)"}
            </div>
          </div>

          {escalation.model_input ? (
            <div>
              <div className="label">Conversation context</div>
              <div className="rounded-md border border-line bg-bg-2/40 px-3 py-2 text-ink-dim max-h-32 overflow-y-auto whitespace-pre-wrap">
                {escalation.model_input}
              </div>
            </div>
          ) : null}

          <div className="text-ink-dim">
            This action paused before reaching the model. Approve to let it
            continue, or Deny to block it.
          </div>
        </div>

        <div className="mt-5 flex justify-end gap-2">
          <button
            type="button"
            className="btn-danger"
            onClick={() => resolve("deny")}
            disabled={busy}
          >
            Deny
          </button>
          <button
            type="button"
            className="btn-primary"
            onClick={() => resolve("approve")}
            disabled={busy}
          >
            Approve
          </button>
        </div>
      </div>
    </div>
  );
}
