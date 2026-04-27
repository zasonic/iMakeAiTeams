import { useEffect, useState } from "react";

import { Settings, type SettingsPayload } from "@/api/client";
import { useAppStore } from "@/stores/appStore";

export function SettingsPanel() {
  const ready = useAppStore((s) => s.sidecarStatus?.status === "ready");
  const pushToast = useAppStore((s) => s.pushToast);
  const [config, setConfig] = useState<SettingsPayload | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [verifying, setVerifying] = useState(false);

  const reload = async () => {
    try {
      setConfig(await Settings.get());
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Could not load settings",
      });
    }
  };

  useEffect(() => {
    if (ready) reload();
  }, [ready]);

  const save = async (key: keyof SettingsPayload, value: unknown) => {
    try {
      await Settings.save(String(key), value);
      reload();
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Save failed",
      });
    }
  };

  const verifyKey = async () => {
    if (!apiKey.trim()) return;
    setVerifying(true);
    try {
      const rsp = await Settings.verifyApiKey(apiKey);
      if (rsp.ok) {
        pushToast({ kind: "success", text: rsp.message });
        setApiKey("");
        reload();
      } else {
        pushToast({ kind: "error", text: rsp.message });
      }
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Verify failed",
      });
    } finally {
      setVerifying(false);
    }
  };

  if (!config) {
    return (
      <div className="p-6 text-ink-dim text-sm">
        {ready ? "Loading…" : "Waiting for backend…"}
      </div>
    );
  }

  return (
    <div className="p-6 overflow-y-auto h-full max-w-2xl space-y-4">
      <header>
        <h1 className="text-xl font-semibold">Settings</h1>
        <p className="text-sm text-ink-dim">API keys, model selection, routing.</p>
      </header>

      <section className="card">
        <h3 className="font-semibold mb-2">Anthropic API key</h3>
        <div className="text-sm text-ink-dim mb-2">
          {config.claude_api_key_set
            ? `Stored in OS keyring · ${config.claude_api_key}`
            : "Not configured."}
        </div>
        <div className="flex gap-2">
          <input
            className="input"
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="sk-ant-…"
          />
          <button className="btn-primary" onClick={verifyKey} disabled={verifying}>
            {verifying ? "Verifying…" : "Verify & save"}
          </button>
        </div>
      </section>

      <section className="card">
        <h3 className="font-semibold mb-2">Model</h3>
        <label className="label">Claude model</label>
        <input
          className="input"
          value={config.claude_model}
          onChange={(e) =>
            setConfig({ ...config, claude_model: e.target.value })
          }
          onBlur={() => save("claude_model", config.claude_model)}
        />
      </section>

      <section className="card">
        <h3 className="font-semibold mb-2">Routing</h3>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={!!config.routing_enabled}
            onChange={(e) => save("routing_enabled", e.target.checked)}
          />
          <span className="text-sm">
            Smart routing (uncertainty-aware classifier picks Claude vs local)
          </span>
        </label>
      </section>

      <section className="card">
        <h3 className="font-semibold mb-2">Local models</h3>
        <label className="label">Ollama URL</label>
        <input
          className="input mb-2"
          value={config.ollama_url}
          onChange={(e) => setConfig({ ...config, ollama_url: e.target.value })}
          onBlur={() => save("ollama_url", config.ollama_url)}
        />
        <label className="label">LM Studio URL</label>
        <input
          className="input"
          value={config.lm_studio_url}
          onChange={(e) =>
            setConfig({ ...config, lm_studio_url: e.target.value })
          }
          onBlur={() => save("lm_studio_url", config.lm_studio_url)}
        />
      </section>

      <section className="card">
        <h3 className="font-semibold mb-2">System prompt</h3>
        <textarea
          className="input min-h-[120px] font-mono text-xs"
          value={config.system_prompt}
          onChange={(e) =>
            setConfig({ ...config, system_prompt: e.target.value })
          }
          onBlur={() => save("system_prompt", config.system_prompt)}
        />
      </section>
    </div>
  );
}
