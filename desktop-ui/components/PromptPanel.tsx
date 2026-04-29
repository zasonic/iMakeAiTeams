import { useEffect, useState } from "react";

import { Prompts } from "@/api/client";
import { useAppStore } from "@/stores/appStore";

interface PromptRow {
  id: string;
  name: string;
  category?: string;
  is_protected?: boolean | number;
}

export function PromptPanel() {
  const ready = useAppStore((s) => s.sidecarStatus?.status === "ready");
  const pushToast = useAppStore((s) => s.pushToast);
  const [prompts, setPrompts] = useState<PromptRow[]>([]);

  useEffect(() => {
    if (!ready) return;
    Prompts.list()
      .then((rows) => setPrompts(rows as PromptRow[]))
      .catch((err) =>
        pushToast({
          kind: "error",
          text: err instanceof Error ? err.message : "Could not load prompts",
        }),
      );
  }, [ready, pushToast]);

  return (
    <div className="p-6 overflow-y-auto h-full">
      <header className="mb-4">
        <h1 className="text-xl font-semibold">Prompt library</h1>
        <p className="text-sm text-ink-dim">
          Versioned system prompts used by the orchestrator. Builtins are
          protected.
        </p>
      </header>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {prompts.map((p) => (
          <div key={p.id} className="card">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold">{p.name}</h3>
              <span className="pill">{p.category ?? "—"}</span>
            </div>
            {p.is_protected ? (
              <span className="pill mt-2">Protected</span>
            ) : null}
          </div>
        ))}
        {!prompts.length && <div className="text-ink-faint text-sm">No prompts.</div>}
      </div>
    </div>
  );
}
