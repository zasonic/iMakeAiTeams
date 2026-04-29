import { useEffect, useState } from "react";

import { System } from "@/api/client";
import { useAppStore } from "@/stores/appStore";

interface ErrorRow {
  id: string;
  timestamp?: string;
  component?: string;
  error_class?: string;
  error_message?: string;
}

export function DiagnosticsPanel() {
  const ready = useAppStore((s) => s.sidecarStatus?.status === "ready");
  const services = useAppStore((s) => s.serviceStatus);
  const setServiceStatus = useAppStore((s) => s.setServiceStatus);
  const pushToast = useAppStore((s) => s.pushToast);
  const [errors, setErrors] = useState<ErrorRow[]>([]);

  useEffect(() => {
    if (!ready) return;
    System.serviceStatus().then(setServiceStatus).catch(() => {});
    System.errorLogs(50)
      .then((rows) => setErrors(rows as ErrorRow[]))
      .catch(() => {});
  }, [ready, setServiceStatus]);

  const runHealth = async () => {
    try {
      await System.runHealthCheck();
      pushToast({ kind: "info", text: "Health check started" });
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Health check failed",
      });
    }
  };

  const exportZip = async () => {
    try {
      await System.exportDiagnostics();
      pushToast({ kind: "info", text: "Diagnostics export started" });
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Export failed",
      });
    }
  };

  const entries = Object.entries(services);

  return (
    <div className="p-6 overflow-y-auto h-full">
      <header className="mb-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Diagnostics</h1>
          <p className="text-sm text-ink-dim">
            Service health, error log, and export bundle.
          </p>
        </div>
        <div className="flex gap-2">
          <button className="btn-ghost" onClick={runHealth} disabled={!ready}>
            Run health check
          </button>
          <button className="btn-primary" onClick={exportZip} disabled={!ready}>
            Export diagnostics
          </button>
        </div>
      </header>

      <section className="card mb-4">
        <h3 className="font-semibold mb-2">Service status</h3>
        <ul className="grid grid-cols-1 md:grid-cols-2 gap-1 text-sm">
          {entries.map(([name, s]) => (
            <li key={name} className="flex items-center gap-2">
              <span
                className={`h-2 w-2 rounded-full ${s.ok ? "bg-ok" : "bg-err"}`}
              />
              <span className="font-mono text-xs">{name}</span>
              {!s.ok && s.error && (
                <span className="text-ink-faint text-xs truncate" title={s.error}>
                  {s.error}
                </span>
              )}
            </li>
          ))}
          {!entries.length && (
            <li className="text-ink-faint">No service report yet.</li>
          )}
        </ul>
      </section>

      <section className="card">
        <h3 className="font-semibold mb-2">Recent errors</h3>
        <ul className="space-y-1 text-sm">
          {errors.map((row) => (
            <li
              key={row.id}
              className="border-b border-line/30 py-1 grid grid-cols-[120px_120px_1fr] gap-2"
            >
              <span className="text-ink-faint">{row.timestamp?.slice(0, 19)}</span>
              <span className="font-mono">{row.component}</span>
              <span className="truncate" title={row.error_message}>
                {row.error_class}: {row.error_message}
              </span>
            </li>
          ))}
          {!errors.length && <li className="text-ink-faint">No errors logged.</li>}
        </ul>
      </section>
    </div>
  );
}
