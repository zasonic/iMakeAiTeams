// src/components/StatusBar.tsx — top-of-app status indicator.
//
// Shows a colored dot for the sidecar lifecycle, the assigned port, the app
// version, and (when crashed) a Restart Backend button.

import { useEffect, useState } from "react";

import { useAppStore } from "@/stores/appStore";

export function StatusBar() {
  const status = useAppStore((s) => s.sidecarStatus);
  const [version, setVersion] = useState<string>("");

  useEffect(() => {
    let alive = true;
    window.electronAPI.getAppVersion().then((v) => {
      if (alive) setVersion(v);
    });
    return () => {
      alive = false;
    };
  }, []);

  const dot = (() => {
    if (!status) return "bg-ink-faint";
    switch (status.status) {
      case "ready":
        return "bg-ok shadow-[0_0_8px_rgba(61,214,140,0.5)]";
      case "starting":
        return "bg-warn animate-pulse";
      case "crashed":
        return "bg-err";
      case "stopped":
        return "bg-ink-faint";
    }
  })();

  const label = (() => {
    if (!status) return "Initializing…";
    switch (status.status) {
      case "ready":
        return `Ready · :${status.port}`;
      case "starting":
        return "Starting backend…";
      case "crashed":
        return status.error || "Backend crashed";
      case "stopped":
        return "Backend stopped";
    }
  })();

  const handleRestart = async () => {
    try {
      await window.electronAPI.restartSidecar();
    } catch (err) {
      console.error("restart failed:", err);
    }
  };

  return (
    <div className="flex items-center justify-between gap-3 px-4 py-2 border-b border-line bg-bg-1/80 backdrop-blur">
      <div className="flex items-center gap-2 min-w-0">
        <span className={`h-2 w-2 rounded-full flex-shrink-0 ${dot}`} aria-hidden />
        <span className="text-xs text-ink-dim truncate" title={label}>
          {label}
        </span>
      </div>
      <div className="flex items-center gap-3">
        {status?.status === "crashed" && (
          <button className="btn-danger text-xs" onClick={handleRestart}>
            Restart Backend
          </button>
        )}
        {version && (
          <span className="text-xs text-ink-faint font-mono">v{version}</span>
        )}
      </div>
    </div>
  );
}
