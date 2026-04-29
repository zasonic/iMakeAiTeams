import { useEffect, useState } from "react";

import { Mcp } from "@/api/client";
import { useAppStore } from "@/stores/appStore";

interface McpServer {
  server_id: string;
  name: string;
  version?: string;
  tool_count: number;
  enabled: boolean;
  env_keys: string[];
  env_set?: Record<string, boolean>;
}

interface McpListResponse {
  servers: McpServer[];
  root: string;
}

export function McpPanel() {
  const ready = useAppStore((s) => s.sidecarStatus?.status === "ready");
  const pushToast = useAppStore((s) => s.pushToast);
  const [data, setData] = useState<McpListResponse | null>(null);

  const refresh = async () => {
    try {
      const rsp = (await Mcp.list()) as McpListResponse;
      setData(rsp);
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "MCP refresh failed",
      });
    }
  };

  useEffect(() => {
    if (ready) refresh();
  }, [ready]);

  const install = async () => {
    const folder = await window.electronAPI.selectFolder();
    if (!folder) return;
    try {
      const rsp = (await Mcp.install(folder)) as { ok: boolean; error?: string };
      if (rsp.ok) {
        pushToast({ kind: "success", text: "MCP server installed" });
        refresh();
      } else {
        pushToast({ kind: "error", text: rsp.error ?? "Install failed" });
      }
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Install failed",
      });
    }
  };

  return (
    <div className="p-6 overflow-y-auto h-full">
      <header className="mb-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">MCP servers</h1>
          <p className="text-sm text-ink-dim">
            {data?.root ? `Storage: ${data.root}` : "Discovering tool servers…"}
          </p>
        </div>
        <button className="btn-primary" onClick={install} disabled={!ready}>
          + Install
        </button>
      </header>

      <div className="grid grid-cols-1 gap-3">
        {(data?.servers ?? []).map((s) => (
          <div key={s.server_id} className="card">
            <div className="flex items-center justify-between mb-1">
              <h3 className="font-semibold">{s.name}</h3>
              <button
                className={`pill ${s.enabled ? "text-ok border-ok/40" : "text-ink-faint"}`}
                onClick={async () => {
                  try {
                    await Mcp.setEnabled(s.server_id, !s.enabled);
                    refresh();
                  } catch (err) {
                    pushToast({
                      kind: "error",
                      text: err instanceof Error ? err.message : "Toggle failed",
                    });
                  }
                }}
              >
                {s.enabled ? "Enabled" : "Disabled"}
              </button>
            </div>
            <div className="text-xs text-ink-dim">
              {s.tool_count} tool{s.tool_count === 1 ? "" : "s"}
              {s.version ? ` · v${s.version}` : ""}
            </div>
            {s.env_keys.length > 0 && (
              <div className="mt-2 text-xs text-ink-faint">
                env: {s.env_keys.join(", ")}
              </div>
            )}
          </div>
        ))}
        {data && !data.servers.length && (
          <div className="text-ink-faint text-sm">No MCP servers installed yet.</div>
        )}
      </div>
    </div>
  );
}
