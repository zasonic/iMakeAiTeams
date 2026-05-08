import { useEffect, useState } from "react";

import { System, type LocalModelRow } from "@/api/client";
import { t } from "@/i18n";
import { useAppStore } from "@/stores/appStore";

function formatBytes(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n) || n <= 0) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v >= 10 || i === 0 ? Math.round(v) : v.toFixed(1)} ${units[i]}`;
}

export function ModelBrowser() {
  const ready = useAppStore((s) => s.sidecarStatus?.status === "ready");
  const pushToast = useAppStore((s) => s.pushToast);
  const [models, setModels] = useState<LocalModelRow[]>([]);
  const [current, setCurrent] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [busyId, setBusyId] = useState<string>("");

  const load = async () => {
    setLoading(true);
    try {
      const res = await System.listLocalModels();
      setModels(res.models);
      setCurrent(res.current);
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Could not load models",
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!ready) return;
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready]);

  const useModel = async (id: string) => {
    setBusyId(id);
    try {
      const res = await System.setActiveLocalModel(id);
      setCurrent(res.current);
      setModels((prev) =>
        prev.map((m) => ({ ...m, loaded: m.id === res.current })),
      );
      pushToast({ kind: "success", text: `Active model set to ${id}` });
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Could not set active model",
      });
    } finally {
      setBusyId("");
    }
  };

  const ollamaCount = models.filter((m) => m.backend === "ollama").length;
  const lmCount = models.filter((m) => m.backend === "lm_studio").length;
  const summary = `${ollamaCount} ${t("ollama")} model${ollamaCount === 1 ? "" : "s"}, ${lmCount} ${t("lm_studio")} model${lmCount === 1 ? "" : "s"}`;

  return (
    <div className="p-6 overflow-y-auto h-full">
      <header className="mb-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Models</h1>
          <p className="text-sm text-ink-dim" data-testid="model-summary">
            {summary}
          </p>
        </div>
        <button
          type="button"
          className="btn-ghost"
          onClick={load}
          disabled={!ready || loading}
        >
          {loading ? "Refreshing…" : "Refresh"}
        </button>
      </header>

      {models.length === 0 ? (
        <div className="card text-sm text-ink-dim" data-testid="model-empty">
          No local models found. Start Ollama or LM Studio, load a model, then
          click Refresh.
        </div>
      ) : (
        <ul className="space-y-2">
          {models.map((m) => {
            const isActive = m.id === current;
            return (
              <li
                key={`${m.backend}:${m.id}`}
                data-testid="model-row"
                className={`card flex items-center justify-between gap-4 ${
                  isActive ? "border-accent/40" : ""
                }`}
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-medium break-all">{m.id}</span>
                    {isActive && (
                      <span
                        data-testid="active-badge"
                        className="text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded bg-accent/15 text-accent"
                      >
                        Active
                      </span>
                    )}
                    <span className="text-[11px] uppercase tracking-wide text-ink-faint">
                      {t(m.backend)}
                    </span>
                    {m.quantization && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-bg-2 text-ink-dim">
                        {m.quantization}
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-ink-faint mt-0.5">
                    {formatBytes(m.size_bytes)}
                    {m.context_length
                      ? ` · ${m.context_length.toLocaleString()} ctx`
                      : ""}
                  </div>
                </div>
                <button
                  type="button"
                  className="btn-primary whitespace-nowrap"
                  onClick={() => useModel(m.id)}
                  disabled={!ready || busyId === m.id || isActive}
                >
                  {isActive
                    ? "In use"
                    : busyId === m.id
                      ? "Setting…"
                      : "Use this model"}
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
