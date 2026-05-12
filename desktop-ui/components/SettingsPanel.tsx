import { useEffect, useState } from "react";

import {
  Chat,
  Docker,
  Models,
  PromptTemplates,
  Settings,
  System,
  Voice,
  type DockerStatus,
  type ModelCatalogEntry,
  type PromptTemplate,
  type SettingsPayload,
  type VoiceAssetsStatus,
} from "@/api/client";
import { t } from "@/i18n";
import { VoiceSetupModal } from "@/components/VoiceSetupModal";
import { useAppStore } from "@/stores/appStore";

interface RouterStats {
  total_exchanges: number;
  error_rate_overall: number;
  by_complexity: Record<string, { total: number; errors: number; error_rate: number }>;
  by_route: Record<string, { total: number; errors: number; error_rate: number }>;
  recent: Array<{
    route: string;
    complexity: string;
    had_error: boolean;
    model_used: string;
    created_at: string;
  }>;
}

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
  const [routerStats, setRouterStats] = useState<RouterStats | null>(null);
  // PR 17: voice setup modal + asset readiness probe.
  const [voiceModalOpen, setVoiceModalOpen] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState<VoiceAssetsStatus | null>(null);
  // Layer B1: Claude model dropdown is sourced from
  // /api/models/catalog so the renderer can't offer an id the backend
  // price math doesn't know about.
  const [modelCatalog, setModelCatalog] = useState<ModelCatalogEntry[]>([]);

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

  const reloadRouterStats = async () => {
    try {
      const stats = (await Chat.routerStats()) as RouterStats;
      setRouterStats(stats);
    } catch {
      setRouterStats(null);
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

  const reloadVoiceStatus = async () => {
    try {
      const status = await Voice.assetsStatus();
      setVoiceStatus(status);
    } catch {
      setVoiceStatus(null);
    }
  };

  const reloadModelCatalog = async () => {
    try {
      const rsp = await Models.catalog();
      setModelCatalog(rsp.models);
    } catch {
      // Fall back to a free-text input if the catalog can't load — better
      // than locking the user out of changing the model.
      setModelCatalog([]);
    }
  };

  useEffect(() => {
    if (ready) {
      reload();
      refreshDocker();
      reloadRouterStats();
      reloadVoiceStatus();
      reloadModelCatalog();
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
      // Only ask the daemon to stop the container if it's actually running;
      // otherwise the round-trip (and any error toast it surfaces via SSE)
      // is just noise on machines without Docker installed.
      if (dockerStatus?.openclaw_running) {
        try {
          await Docker.stop();
        } catch {
          /* surfaced via SSE */
        }
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

  // PR 17: voice toggle. When the user enables either voice flag and the
  // backend reports the assets aren't ready yet, fire the setup modal so
  // they don't get a confusing 503 the first time they hit the mic.
  const toggleVoiceInput = async (enabled: boolean) => {
    await save("voice_input_enabled", enabled);
    if (enabled) {
      const fresh = await Voice.assetsStatus().catch(() => null);
      setVoiceStatus(fresh);
      if (fresh && !fresh.stt_ready) {
        setVoiceModalOpen(true);
      }
    }
  };

  const toggleVoiceOutput = async (enabled: boolean) => {
    await save("voice_output_enabled", enabled);
    if (enabled) {
      const fresh = await Voice.assetsStatus().catch(() => null);
      setVoiceStatus(fresh);
      if (fresh && !fresh.tts_ready) {
        setVoiceModalOpen(true);
      }
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
        {modelCatalog.length > 0 ? (
          <select
            className="input"
            value={config.claude_model}
            onChange={(e) => {
              const next = e.target.value;
              setConfig({ ...config, claude_model: next });
              save("claude_model", next);
            }}
          >
            {/* If the saved model isn't in the catalog (e.g. older
                install, custom id), keep it visible at the top so the
                user isn't silently switched. */}
            {modelCatalog.find((m) => m.id === config.claude_model)
              ? null
              : (
                <option value={config.claude_model}>
                  {config.claude_model} (not in catalog)
                </option>
              )}
            {modelCatalog.map((m) => (
              <option key={m.id} value={m.id}>
                {m.display_name} — ${m.input_price_per_mtok}/${m.output_price_per_mtok} per MTok
              </option>
            ))}
          </select>
        ) : (
          // Catalog failed to load — fall back to the legacy free-text
          // input so a transient API failure doesn't strand the user.
          <input
            className="input"
            value={config.claude_model}
            onChange={(e) =>
              setConfig({ ...config, claude_model: e.target.value })
            }
            onBlur={() => save("claude_model", config.claude_model)}
          />
        )}
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

      <RoutingPerformanceSection stats={routerStats} />

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
        <h3 className="font-semibold mb-2">Updates</h3>
        <label className="flex items-start gap-2">
          <input
            type="checkbox"
            className="mt-1"
            checked={!!config.auto_update_enabled}
            onChange={(e) => save("auto_update_enabled", e.target.checked)}
          />
          <span className="text-sm">
            <div>Automatic updates</div>
            <div className="text-xs text-ink-dim">
              Check for new versions in the background. Never restarts without
              your permission.
            </div>
          </span>
        </label>
      </section>

      <section className="card" data-testid="settings-voice-section">
        <h3 className="font-semibold mb-2">Voice</h3>
        <p className="text-sm text-ink-dim mb-3">
          Speak to dictate messages and have replies read back to you. Both
          features run on your machine — audio never leaves this computer.
        </p>
        <label className="flex items-start gap-2 mb-2">
          <input
            type="checkbox"
            data-testid="settings-voice-input-toggle"
            className="mt-1"
            checked={!!config.voice_input_enabled}
            onChange={(e) => toggleVoiceInput(e.target.checked)}
          />
          <span className="text-sm">
            <div>Voice input</div>
            <div className="text-xs text-ink-dim">
              Adds a mic button to the chat input. Click to record, click again
              to transcribe with Whisper.cpp.
            </div>
          </span>
        </label>
        <label className="flex items-start gap-2 mb-3">
          <input
            type="checkbox"
            data-testid="settings-voice-output-toggle"
            className="mt-1"
            checked={!!config.voice_output_enabled}
            onChange={(e) => toggleVoiceOutput(e.target.checked)}
          />
          <span className="text-sm">
            <div>Voice output</div>
            <div className="text-xs text-ink-dim">
              Adds a speaker button to assistant messages. Click to hear the
              reply spoken aloud with Piper.
            </div>
          </span>
        </label>

        {(config.voice_input_enabled || config.voice_output_enabled) && (
          <>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="label">Speech-to-text model</label>
                <input
                  className="input"
                  value={config.stt_model_id || ""}
                  onChange={(e) =>
                    setConfig({ ...config, stt_model_id: e.target.value })
                  }
                  onBlur={() => save("stt_model_id", config.stt_model_id)}
                />
              </div>
              <div>
                <label className="label">Text-to-speech voice</label>
                <input
                  className="input"
                  value={config.tts_voice_id || ""}
                  onChange={(e) =>
                    setConfig({ ...config, tts_voice_id: e.target.value })
                  }
                  onBlur={() => save("tts_voice_id", config.tts_voice_id)}
                />
              </div>
            </div>

            <div className="rounded-md border border-line bg-bg-2 px-3 py-2 text-xs space-y-1.5 mt-3">
              <div className="flex items-center justify-between">
                <span className="text-ink-dim">STT model</span>
                <span className={voiceStatus?.stt_ready ? "text-ok" : "text-warn"}>
                  {voiceStatus == null
                    ? "Checking…"
                    : voiceStatus.stt_ready
                      ? "Ready"
                      : "Not downloaded"}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-ink-dim">TTS voice</span>
                <span className={voiceStatus?.tts_ready ? "text-ok" : "text-warn"}>
                  {voiceStatus == null
                    ? "Checking…"
                    : voiceStatus.tts_ready
                      ? "Ready"
                      : "Not downloaded"}
                </span>
              </div>
              <div className="flex gap-2 pt-2">
                <button
                  type="button"
                  className="btn-ghost text-xs"
                  onClick={reloadVoiceStatus}
                >
                  Re-check
                </button>
                {(voiceStatus && (!voiceStatus.stt_ready || !voiceStatus.tts_ready)) && (
                  <button
                    type="button"
                    className="btn-primary text-xs"
                    onClick={() => setVoiceModalOpen(true)}
                    data-testid="settings-voice-download"
                  >
                    Download models
                  </button>
                )}
              </div>
            </div>
          </>
        )}
      </section>

      <VoiceSetupModal
        open={voiceModalOpen}
        onClose={() => setVoiceModalOpen(false)}
        onComplete={() => {
          setVoiceModalOpen(false);
          reloadVoiceStatus();
        }}
      />

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
        <SystemPromptTemplatesPicker
          onApply={(tmpl) => {
            setConfig({ ...config, system_prompt: tmpl.body });
            save("system_prompt", tmpl.body);
          }}
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

      <section className="card">
        <h3 className="font-semibold mb-2">Troubleshooting</h3>
        <div className="space-y-2">
          <button
            type="button"
            className="btn-ghost"
            onClick={async () => {
              try {
                await System.exportDiagnostics();
                pushToast({ kind: "success", text: "Diagnostics exported" });
              } catch {
                pushToast({ kind: "error", text: "Export failed" });
              }
            }}
          >
            Export diagnostics
          </button>
          <button
            type="button"
            className="btn-danger"
            onClick={() => {
              const ok = window.confirm(
                "This will clear your conversation history, memory, and settings. Your API key (stored in the OS keyring) will not be affected. This cannot be undone. Continue?",
              );
              if (ok) {
                window.electronAPI.restartSidecar();
              }
            }}
          >
            Reset to defaults
          </button>
        </div>
      </section>
    </div>
  );
}

// ── PR 18: pick a saved system_prompt template as the default ──────────────

interface SystemPromptTemplatesPickerProps {
  onApply: (tmpl: PromptTemplate) => void;
}

function SystemPromptTemplatesPicker({
  onApply,
}: SystemPromptTemplatesPickerProps) {
  const ready = useAppStore((s) => s.sidecarStatus?.status === "ready");
  const pushToast = useAppStore((s) => s.pushToast);
  const templates = useAppStore((s) => s.promptTemplates);
  const setPromptTemplates = useAppStore((s) => s.setPromptTemplates);

  useEffect(() => {
    if (!ready) return;
    let alive = true;
    PromptTemplates.list()
      .then((rows) => {
        if (alive) setPromptTemplates(rows);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [ready, setPromptTemplates]);

  const systemPrompts = templates.filter((p) => p.kind === "system_prompt");
  if (systemPrompts.length === 0) return null;

  return (
    <div
      className="mt-3 rounded-md border border-line bg-bg-2 px-3 py-2"
      data-testid="settings-system-prompt-templates"
    >
      <div className="text-xs text-ink-dim mb-2">
        {t("prompts.set_as_default_system_prompt")}
      </div>
      <ul className="space-y-1">
        {systemPrompts.map((tmpl) => (
          <li
            key={tmpl.id}
            className="flex items-center justify-between gap-2 text-xs"
          >
            <span className="truncate">{tmpl.title}</span>
            <button
              type="button"
              className="btn-ghost text-xs"
              data-testid={`settings-set-default-${tmpl.id}`}
              onClick={() => {
                onApply(tmpl);
                pushToast({
                  kind: "success",
                  text: t("prompts.set_as_default.success"),
                });
              }}
            >
              {t("prompts.set_as_default_system_prompt")}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function RoutingPerformanceSection({ stats }: { stats: RouterStats | null }) {
  if (!stats || stats.total_exchanges === 0) {
    return (
      <section className="card">
        <h3 className="font-semibold mb-2">Routing Performance</h3>
        <div className="text-sm text-ink-dim">No routing data yet</div>
      </section>
    );
  }

  const errorPct = (stats.error_rate_overall * 100).toFixed(1);
  const claudeCount = stats.by_route?.claude?.total ?? 0;
  const localCount = stats.by_route?.local?.total ?? 0;
  const complexityKeys = ["simple", "medium", "complex"];

  return (
    <section className="card">
      <h3 className="font-semibold mb-2">Routing Performance</h3>
      <div className="grid grid-cols-3 gap-2 mb-3">
        <div className="text-center">
          <div className="text-lg font-semibold">{stats.total_exchanges}</div>
          <div className="text-xs text-ink-dim">Total</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-semibold">{errorPct}%</div>
          <div className="text-xs text-ink-dim">Error rate</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-semibold">
            {claudeCount} / {localCount}
          </div>
          <div className="text-xs text-ink-dim">Claude / local</div>
        </div>
      </div>

      <div className="text-xs font-semibold mb-1">By complexity</div>
      <div className="grid grid-cols-3 gap-2 mb-3 text-xs">
        {complexityKeys.map((k) => {
          const b = stats.by_complexity?.[k];
          return (
            <div key={k} className="border border-base-700 rounded p-2">
              <div className="capitalize text-ink-dim">{k}</div>
              <div>
                {b ? `${b.total} · ${(b.error_rate * 100).toFixed(1)}% err` : "—"}
              </div>
            </div>
          );
        })}
      </div>

      {stats.recent.length > 0 && (
        <>
          <div className="text-xs font-semibold mb-1">Recent decisions</div>
          <div className="text-xs space-y-1 max-h-48 overflow-y-auto">
            {stats.recent.slice(0, 10).map((r, i) => (
              <div
                key={i}
                className="flex justify-between gap-2 border-b border-base-700/50 pb-1"
              >
                <span className="text-ink-dim">{r.route}</span>
                <span className="text-ink-dim">{r.complexity}</span>
                <span className={r.had_error ? "text-rose-400" : ""}>
                  {r.model_used || "—"}
                </span>
              </div>
            ))}
          </div>
        </>
      )}
    </section>
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
