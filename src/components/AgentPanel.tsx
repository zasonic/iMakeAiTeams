import { useEffect, useState } from "react";

import { Agents } from "@/api/client";
import { useAppStore } from "@/stores/appStore";

interface AgentRow {
  id: string;
  name: string;
  description?: string;
  model_preference?: string;
  is_builtin?: boolean | number;
}

export function AgentPanel() {
  const ready = useAppStore((s) => s.sidecarStatus?.status === "ready");
  const pushToast = useAppStore((s) => s.pushToast);
  const [agents, setAgents] = useState<AgentRow[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!ready) return;
    let alive = true;
    Agents.list()
      .then((rows) => {
        if (alive) setAgents(rows as AgentRow[]);
      })
      .catch((err) => {
        if (alive) pushToast({ kind: "error", text: err.message });
      })
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [ready, pushToast]);

  return (
    <div className="p-6 overflow-y-auto h-full">
      <header className="mb-4">
        <h1 className="text-xl font-semibold">Agents</h1>
        <p className="text-sm text-ink-dim">
          Define personas, model preferences, and budgets. Builtin agents
          can't be deleted.
        </p>
      </header>
      {loading && <div className="text-ink-faint text-sm">Loading…</div>}
      {!loading && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {agents.map((a) => (
            <div key={a.id} className="card">
              <div className="flex items-center justify-between mb-1">
                <h3 className="font-semibold">{a.name}</h3>
                {a.is_builtin ? (
                  <span className="pill">Builtin</span>
                ) : (
                  <span className="pill text-accent border-accent/40">Custom</span>
                )}
              </div>
              <p className="text-sm text-ink-dim mb-2 line-clamp-3">{a.description}</p>
              {a.model_preference && (
                <span className="pill">model: {a.model_preference}</span>
              )}
            </div>
          ))}
          {!agents.length && (
            <div className="text-ink-faint text-sm">No agents found.</div>
          )}
        </div>
      )}
    </div>
  );
}
