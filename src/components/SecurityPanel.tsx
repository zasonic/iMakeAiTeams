import { useEffect, useState } from "react";

import { System } from "@/api/client";
import { useAppStore } from "@/stores/appStore";

interface SecurityStatus {
  firewall_enabled: boolean;
  scan_count?: number;
  recent_blocks?: number;
}

interface ScanRow {
  id: string;
  timestamp?: string;
  verdict?: string;
  score?: number;
}

export function SecurityPanel() {
  const ready = useAppStore((s) => s.sidecarStatus?.status === "ready");
  const pushToast = useAppStore((s) => s.pushToast);
  const [status, setStatus] = useState<SecurityStatus | null>(null);
  const [log, setLog] = useState<ScanRow[]>([]);

  useEffect(() => {
    if (!ready) return;
    Promise.all([System.securityStatus(), System.scanLog(50)])
      .then(([s, rows]) => {
        setStatus(s as SecurityStatus);
        setLog(rows as ScanRow[]);
      })
      .catch((err) => pushToast({ kind: "error", text: err.message }));
  }, [ready, pushToast]);

  const toggle = async () => {
    if (!status) return;
    try {
      await System.toggleFirewall(!status.firewall_enabled);
      const next = (await System.securityStatus()) as SecurityStatus;
      setStatus(next);
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Toggle failed",
      });
    }
  };

  return (
    <div className="p-6 overflow-y-auto h-full">
      <header className="mb-4">
        <h1 className="text-xl font-semibold">Security</h1>
        <p className="text-sm text-ink-dim">
          Structural defenses (input firewall, scan log, risk ledger).
        </p>
      </header>
      <div className="card mb-4 flex items-center justify-between">
        <div>
          <div className="font-semibold">Input firewall</div>
          <div className="text-xs text-ink-dim">
            Scans every user message for prompt-injection patterns.
          </div>
        </div>
        <button
          className={status?.firewall_enabled ? "btn-danger" : "btn-primary"}
          onClick={toggle}
          disabled={!ready || !status}
        >
          {status?.firewall_enabled ? "Disable" : "Enable"}
        </button>
      </div>

      <div className="card">
        <h3 className="font-semibold mb-2">Recent scans</h3>
        <ul className="space-y-1 text-sm">
          {log.map((row) => (
            <li
              key={row.id}
              className="flex items-center justify-between border-b border-line/30 py-1"
            >
              <span>{row.timestamp?.slice(0, 19) ?? "—"}</span>
              <span className="pill">{row.verdict ?? "?"}</span>
              <span className="font-mono text-xs">
                {typeof row.score === "number" ? row.score.toFixed(2) : "—"}
              </span>
            </li>
          ))}
          {!log.length && <li className="text-ink-faint">No scans yet.</li>}
        </ul>
      </div>
    </div>
  );
}
