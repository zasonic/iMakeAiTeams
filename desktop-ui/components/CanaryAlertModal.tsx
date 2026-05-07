import { useState } from "react";

import { System } from "@/api/client";
import { useAppStore } from "@/stores/appStore";

export function CanaryAlertModal() {
  const alert = useAppStore((s) => s.canaryAlert);
  const setCanaryAlert = useAppStore((s) => s.setCanaryAlert);
  const setCanaryAlertOpen = useAppStore((s) => s.setCanaryAlertOpen);
  const pushToast = useAppStore((s) => s.pushToast);
  const [resetting, setResetting] = useState(false);

  if (!alert) return null;

  const close = () => setCanaryAlertOpen(false);

  const reset = async () => {
    setResetting(true);
    try {
      const rsp = await System.resetCanary(alert.model_id);
      if (rsp.ok) {
        pushToast({
          kind: "success",
          text: `Baseline cleared for ${alert.model_id}. The canary will re-capture on the next load.`,
        });
        setCanaryAlert(null);
      } else {
        pushToast({
          kind: "error",
          text: rsp.error ?? "Failed to reset baseline.",
        });
      }
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Network error",
      });
    } finally {
      setResetting(false);
    }
  };

  const driftPct = (alert.mean_drift * 100).toFixed(1);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="card max-w-lg w-full mx-4 p-5">
        <div className="flex items-start justify-between gap-3 mb-3">
          <div>
            <h2 className="text-lg font-semibold">Model behavior changed</h2>
            <p className="text-xs text-ink-faint mt-1">
              {alert.model_id}
            </p>
          </div>
          <span className="text-xs uppercase tracking-wide text-warn">
            Canary
          </span>
        </div>

        <div className="border border-line rounded-md p-3 bg-bg-2/40 mb-3">
          <div className="text-[11px] uppercase text-ink-faint mb-1">
            Mean cosine drift
          </div>
          <div className="text-2xl font-semibold text-warn">{driftPct}%</div>
          <div className="text-xs text-ink-dim mt-1">
            Threshold: 40.0%. Higher values mean the model's responses have
            diverged from the baseline captured on first load.
          </div>
        </div>

        {alert.drifted_prompts.length > 0 && (
          <div className="mb-4">
            <div className="text-[11px] uppercase text-ink-faint mb-1">
              Most affected prompts
            </div>
            <ul className="space-y-1">
              {alert.drifted_prompts.slice(0, 3).map((p, i) => (
                <li
                  key={`${i}-${p.slice(0, 24)}`}
                  className="text-sm border border-line rounded-md p-2 bg-bg-2/40 break-words"
                >
                  {p}
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="flex justify-end gap-2">
          <button type="button" className="btn-ghost" onClick={close}>
            Close
          </button>
          <button
            type="button"
            className="btn-primary"
            onClick={reset}
            disabled={resetting}
          >
            {resetting ? "Resetting…" : "Reset baseline"}
          </button>
        </div>
      </div>
    </div>
  );
}
