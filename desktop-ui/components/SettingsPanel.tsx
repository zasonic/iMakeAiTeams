import { useEffect, useState } from "react";

import {
  Docker,
  Settings,
  type DockerStatus,
  type SettingsPayload,
} from "@/api/client";
import { useAppStore } from "@/stores/appStore";

const DOCKER_INSTALL_URL = "https://www.docker.com/products/docker-desktop/";

export function SettingsPanel() {
  const ready = useAppStore((s) => s.sidecarStatus?.status === "ready");
  const pushToast = useAppStore((s) => s.pushToast);
  const dockerStatus = useAppStore((s) => s.dockerStatus);
  const setDockerStatus = useAppStore((s) => s.setDockerStatus);
  const setPowerModeEnabled = useAppStore((s) => s.setPowerModeEnabled);
  const [config, setConfig] = useState<SettingsPayload | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [verifying, setVerifying] = useState(false);
  const [pmApiKey, setPmApiKey] = useState("");
  const [pmBusy, setPmBusy] = useState<"start" | "stop" | "check" | null>(null);

  const reload = async () => {
    try {
      const fresh = await Settings.get();
      setConfig(fresh);
      // Mirror the toggle into the store so the StatusBar / ChatView pick up
      // changes without each having to fetch /api/settings on a timer.
      setPowerModeEnabled(!!fresh.power_mode_enabled);
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Could not load settings",
      });
    }
  };

  const refreshDocker = async () => {
    try {
      const status = await Docker.status();
      setDockerStatus(status);
      return status;
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Could not check Docker",
      });
      return null;
    }
  };

  useEffect(() => {
    if (ready) {
      reload();
      refreshDocker();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
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

  const togglePowerMode = async (enabled: boolean) => {
    await save("power_mode_enabled", enabled);
    if (!enabled) {
      // Stopping the container is safe even if it isn't running.
      try {
        await Docker.stop();
      } catch {
        /* surfaced via SSE */
      }
      await refreshDocker();
      return;
    }
    const status = await refreshDocker();
    if (!status) return;
    if (!status.docker_installed || !status.docker_running) return;
    if (status.openclaw_healthy) return;
    await startOpenclaw();
  };

  const startOpenclaw = async () => {
    setPmBusy("start");
    try {
      const r = await Docker.start();
      if (r.ok) {
        pushToast({ kind: "success", text: "OpenClaw is ready." });
      }
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Could not start OpenClaw",
      });
    } finally {
      await refreshDocker();
      setPmBusy(null);
    }
  };

  const stopOpenclaw = async () => {
    setPmBusy("stop");
    try {
      await Docker.stop();
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Could not stop OpenClaw",
      });
    } finally {
      await refreshDocker();
      setPmBusy(null);
    }
  };

  const recheckDocker = async () => {
    setPmBusy("check");
    try {
      await refreshDocker();
    } finally {
      setPmBusy(null);
    }
  };

  const pickWorkspace = async () => {
    const folder = await window.electronAPI.selectWorkspaceFolder();
    if (folder) {
      await save("power_mode_workspace", folder);
      await refreshDocker();
    }
  };

  const savePmApiKey = async () => {
    if (!pmApiKey.trim()) return;
    await save("power_mode_api_key", pmApiKey);
    setPmApiKey("");
    pushToast({ kind: "success", text: "Power Mode API key saved." });
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

      <PowerModeSection
        config={config}
        dockerStatus={dockerStatus}
        pmApiKey={pmApiKey}
        pmBusy={pmBusy}
        setPmApiKey={setPmApiKey}
        setConfig={setConfig}
        save={save}
        togglePowerMode={togglePowerMode}
        startOpenclaw={startOpenclaw}
        stopOpenclaw={stopOpenclaw}
        recheckDocker={recheckDocker}
        pickWorkspace={pickWorkspace}
        savePmApiKey={savePmApiKey}
      />
    </div>
  );
}

interface PowerModeSectionProps {
  config: SettingsPayload;
  dockerStatus: DockerStatus | null;
  pmApiKey: string;
  pmBusy: "start" | "stop" | "check" | null;
  setPmApiKey: (v: string) => void;
  setConfig: (c: SettingsPayload) => void;
  save: (key: keyof SettingsPayload, value: unknown) => Promise<void>;
  togglePowerMode: (enabled: boolean) => Promise<void>;
  startOpenclaw: () => Promise<void>;
  stopOpenclaw: () => Promise<void>;
  recheckDocker: () => Promise<void>;
  pickWorkspace: () => Promise<void>;
  savePmApiKey: () => Promise<void>;
}

function PowerModeSection({
  config,
  dockerStatus,
  pmApiKey,
  pmBusy,
  setPmApiKey,
  setConfig,
  save,
  togglePowerMode,
  startOpenclaw,
  stopOpenclaw,
  recheckDocker,
  pickWorkspace,
  savePmApiKey,
}: PowerModeSectionProps) {
  const enabled = !!config.power_mode_enabled;
  const dockerReady = !!dockerStatus?.docker_installed && !!dockerStatus?.docker_running;
  const openclawReady = !!dockerStatus?.openclaw_healthy;

  return (
    <section className="card border-accent/30">
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-semibold">Power Mode</h3>
        <span className="text-[10px] uppercase tracking-wide text-ink-faint">v3 · opt-in</span>
      </div>
      <p className="text-sm text-ink-dim mb-3">
        Delegate execution tasks (write code, run shell commands, manage files,
        browse the web) to OpenClaw running in Docker. Chat keeps working
        normally; Power Mode only kicks in for messages classified as
        execution.
      </p>

      <label className="flex items-center gap-2 mb-3">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => togglePowerMode(e.target.checked)}
        />
        <span className="text-sm">Enable Power Mode</span>
      </label>

      {enabled && (
        <>
          <DockerStatusRow
            status={dockerStatus}
            onRecheck={recheckDocker}
            onStart={startOpenclaw}
            onStop={stopOpenclaw}
            busy={pmBusy}
          />

          {!dockerReady && (
            <div className="rounded-md border border-warn/40 bg-warn/5 px-3 py-2 text-xs text-ink mt-3 space-y-1">
              <div className="font-semibold text-warn">Docker is required for Power Mode</div>
              <p className="text-ink-dim">
                Docker lets the AI safely run code and manage files in an
                isolated environment. Install Docker Desktop, start it, then
                click Re-check below.
              </p>
              <button
                type="button"
                className="text-accent hover:underline"
                onClick={() => window.electronAPI.openExternal(DOCKER_INSTALL_URL)}
              >
                Download Docker Desktop →
              </button>
            </div>
          )}

          <div className="space-y-3 mt-4">
            <div>
              <label className="label">Workspace folder</label>
              <div className="flex gap-2">
                <input
                  className="input flex-1"
                  readOnly
                  value={
                    config.power_mode_workspace ||
                    dockerStatus?.workspace_dir ||
                    "(default: ~/Documents/iMakeAiTeams-Workspace)"
                  }
                />
                <button type="button" className="btn-ghost" onClick={pickWorkspace}>
                  Choose…
                </button>
              </div>
              <p className="text-[11px] text-ink-faint mt-1">
                The only host folder OpenClaw can read or write.
              </p>
            </div>

            <div>
              <label className="label">Model provider</label>
              <select
                className="input"
                value={config.power_mode_model_provider}
                onChange={(e) => {
                  setConfig({ ...config, power_mode_model_provider: e.target.value });
                  save("power_mode_model_provider", e.target.value);
                }}
              >
                <option value="anthropic">Anthropic (Claude)</option>
                <option value="openai">OpenAI</option>
                <option value="local">Local (LiteLLM)</option>
              </select>
            </div>

            <div>
              <label className="label">Execution model</label>
              <input
                className="input"
                value={config.power_mode_model_name}
                onChange={(e) =>
                  setConfig({ ...config, power_mode_model_name: e.target.value })
                }
                onBlur={() => save("power_mode_model_name", config.power_mode_model_name)}
              />
            </div>

            <div>
              <label className="label">Provider API key</label>
              <div className="flex gap-2">
                <input
                  className="input flex-1"
                  type="password"
                  value={pmApiKey}
                  onChange={(e) => setPmApiKey(e.target.value)}
                  placeholder={
                    config.power_mode_api_key_set
                      ? `${config.power_mode_api_key} (saved in keyring)`
                      : "Stored in OS keyring"
                  }
                />
                <button
                  type="button"
                  className="btn-primary"
                  onClick={savePmApiKey}
                  disabled={!pmApiKey.trim()}
                >
                  Save
                </button>
              </div>
            </div>

            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={!!config.power_mode_autostart}
                onChange={(e) => save("power_mode_autostart", e.target.checked)}
              />
              <span className="text-sm">Start Power Mode when the app launches</span>
            </label>

            <div>
              <label className="label">Gateway port</label>
              <input
                type="number"
                className="input"
                value={config.power_mode_gateway_port}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    power_mode_gateway_port: Number(e.target.value),
                  })
                }
                onBlur={() =>
                  save("power_mode_gateway_port", config.power_mode_gateway_port)
                }
              />
              <p className="text-[11px] text-ink-faint mt-1">
                Bound to 127.0.0.1 only — never exposed off this machine.
              </p>
            </div>
          </div>

          {dockerReady && openclawReady && (
            <div className="mt-4 text-xs text-ok">
              ⚡ Power Mode is active. Execution-class messages will route through
              OpenClaw automatically.
            </div>
          )}
        </>
      )}
    </section>
  );
}

interface DockerStatusRowProps {
  status: DockerStatus | null;
  onRecheck: () => Promise<void>;
  onStart: () => Promise<void>;
  onStop: () => Promise<void>;
  busy: "start" | "stop" | "check" | null;
}

function DockerStatusRow({ status, onRecheck, onStart, onStop, busy }: DockerStatusRowProps) {
  const dockerLabel = !status
    ? "Checking…"
    : !status.docker_installed
      ? "Not installed"
      : !status.docker_running
        ? "Installed · not running"
        : "Running";
  const openclawLabel = !status
    ? "Checking…"
    : !status.openclaw_running
      ? "Stopped"
      : status.openclaw_healthy
        ? "Ready"
        : "Starting…";

  const dockerTone = status?.docker_running ? "text-ok" : status?.docker_installed ? "text-warn" : "text-err";
  const openclawTone = status?.openclaw_healthy
    ? "text-ok"
    : status?.openclaw_running
      ? "text-warn"
      : "text-ink-faint";

  return (
    <div className="rounded-md border border-line bg-bg-2 px-3 py-2 text-xs space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-ink-dim">Docker</span>
        <span className={dockerTone}>{dockerLabel}</span>
      </div>
      <div className="flex items-center justify-between">
        <span className="text-ink-dim">OpenClaw</span>
        <span className={openclawTone}>{openclawLabel}</span>
      </div>
      {status?.last_error && (
        <div className="text-err text-[11px] pt-1">{status.last_error}</div>
      )}
      {status?.detail && !status.last_error && (
        <div className="text-ink-faint text-[11px] pt-1">{status.detail}</div>
      )}
      <div className="flex gap-2 pt-2">
        <button
          type="button"
          className="btn-ghost text-xs"
          onClick={onRecheck}
          disabled={busy === "check"}
        >
          {busy === "check" ? "Checking…" : "Re-check"}
        </button>
        {status?.docker_running && !status.openclaw_healthy && (
          <button
            type="button"
            className="btn-primary text-xs"
            onClick={onStart}
            disabled={busy === "start"}
          >
            {busy === "start" ? "Starting…" : "Start OpenClaw"}
          </button>
        )}
        {status?.openclaw_running && (
          <button
            type="button"
            className="btn-ghost text-xs"
            onClick={onStop}
            disabled={busy === "stop"}
          >
            {busy === "stop" ? "Stopping…" : "Stop OpenClaw"}
          </button>
        )}
      </div>
    </div>
  );
}
