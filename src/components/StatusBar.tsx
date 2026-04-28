// src/components/StatusBar.tsx — top-of-app status indicator.
//
// Shows a colored dot for the sidecar lifecycle, the assigned port, the app
// version, a Power Mode badge when active, and (when crashed) a Restart
// Backend button.

import { useEffect, useState } from "react";

import { Settings, Docker } from "@/api/client";
import { useAppStore } from "@/stores/appStore";

export function StatusBar() {
  const status = useAppStore((s) => s.sidecarStatus);
  const dockerStatus = useAppStore((s) => s.dockerStatus);
  const setDockerStatus = useAppStore((s) => s.setDockerStatus);
  const [version, setVersion] = useState<string>("");
  const [powerModeEnabled, setPowerModeEnabled] = useState(false);

  useEffect(() => {
    let alive = true;
    window.electronAPI.getAppVersion().then((v) => {
      if (alive) setVersion(v);
    });
    return () => {
      alive = false;
    };
  }, []);

  // Track the Power Mode flag + Docker status so the badge reflects reality.
  useEffect(() => {
    if (status?.status !== "ready") return;
    let alive = true;
    Settings.get()
      .then((s) => {
        if (alive) setPowerModeEnabled(!!s.power_mode_enabled);
      })
      .catch(() => {});
    Docker.status()
      .then((s) => {
        if (alive) setDockerStatus(s);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [status, setDockerStatus]);

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

  const powerModeBadge = (() => {
    if (!powerModeEnabled) return null;
    if (dockerStatus?.openclaw_healthy) {
      return (
        <span
          className="text-[11px] px-1.5 py-0.5 rounded border border-accent/40 bg-accent/10 text-accent"
          title={`OpenClaw ready · ${dockerStatus.gateway_url}`}
        >
          ⚡ Power Mode
        </span>
      );
    }
    if (dockerStatus?.openclaw_running) {
      return (
        <span
          className="text-[11px] px-1.5 py-0.5 rounded border border-warn/40 bg-warn/10 text-warn"
          title="OpenClaw is starting…"
        >
          ⚡ Power Mode · starting
        </span>
      );
    }
    return (
      <span
        className="text-[11px] px-1.5 py-0.5 rounded border border-warn/40 bg-warn/10 text-warn"
        title={dockerStatus?.detail ?? "Power Mode is enabled but OpenClaw isn't running"}
      >
        ⚡ Power Mode · offline
      </span>
    );
  })();

  return (
    <div className="flex items-center justify-between gap-3 px-4 py-2 border-b border-line bg-bg-1/80 backdrop-blur">
      <div className="flex items-center gap-2 min-w-0">
        <span className={`h-2 w-2 rounded-full flex-shrink-0 ${dot}`} aria-hidden />
        <span className="text-xs text-ink-dim truncate" title={label}>
          {label}
        </span>
      </div>
      <div className="flex items-center gap-3">
        {powerModeBadge}
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
