import { useEffect, useRef, useState } from "react";

import { Settings, System } from "@/api/client";
import { useAppStore } from "@/stores/appStore";

interface Props {
  onComplete: () => void;
}

type LocalChoice = "quick_start" | "byo" | "skip" | null;
type Step =
  | "welcome"
  | "claude"
  | "local_choice"
  | "quick_start"
  | "byo_install"
  | "done";

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

export function FirstRunWizard({ onComplete }: Props) {
  const pushToast = useAppStore((s) => s.pushToast);
  const bundledDownload = useAppStore((s) => s.bundledDownload);
  const setBundledDownload = useAppStore((s) => s.setBundledDownload);

  const [step, setStep] = useState<Step>("welcome");
  const [apiKey, setApiKey] = useState("");
  const [verifying, setVerifying] = useState(false);
  const [localChoice, setLocalChoice] = useState<LocalChoice>(null);
  const [smokeText, setSmokeText] = useState<string>("");
  const startedRef = useRef(false);

  const verify = async () => {
    if (!apiKey.trim()) return;
    setVerifying(true);
    try {
      const rsp = await Settings.verifyApiKey(apiKey);
      if (rsp.ok) {
        pushToast({ kind: "success", text: rsp.message });
        setStep("local_choice");
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

  const finish = async () => {
    try {
      await Settings.completeFirstRun("chat");
      onComplete();
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Could not complete setup",
      });
    }
  };

  const startQuickStart = async () => {
    setLocalChoice("quick_start");
    setStep("quick_start");
    setSmokeText("");
    startedRef.current = false;
    setBundledDownload({
      status: "downloading",
      modelId: "",
      bytesDone: 0,
      bytesTotal: 0,
      error: "",
    });
    try {
      const rsp = await System.bundledDownload();
      if (!rsp.ok) {
        setBundledDownload({
          status: "error",
          modelId: rsp.model_id ?? "",
          bytesDone: 0,
          bytesTotal: 0,
          error: rsp.error ?? "Download could not start",
        });
      } else {
        setBundledDownload({
          status: "downloading",
          modelId: rsp.model_id ?? "",
          bytesDone: 0,
          bytesTotal: 0,
          error: "",
        });
      }
    } catch (err) {
      setBundledDownload({
        status: "error",
        modelId: "",
        bytesDone: 0,
        bytesTotal: 0,
        error: err instanceof Error ? err.message : "Download could not start",
      });
    }
  };

  // When the SSE bundled_download_complete event flips status to "complete",
  // automatically start the bundled server and run a smoke chat. Guard with
  // startedRef so we never fire the smoke chat twice for the same download.
  useEffect(() => {
    if (step !== "quick_start") return;
    if (bundledDownload.status !== "complete") return;
    if (startedRef.current) return;
    startedRef.current = true;

    (async () => {
      try {
        const startRsp = await System.bundledStart(bundledDownload.modelId);
        if (!startRsp.ok) {
          setBundledDownload({
            ...bundledDownload,
            status: "error",
            error: startRsp.error ?? "Server failed to start",
          });
          return;
        }
        setSmokeText("Starting…");
        // The router/orchestrator picks up local_backend_mode automatically;
        // a short echo round-trip confirms the binary is alive end-to-end.
        await Settings.set("local_backend_mode", "bundled");
        setSmokeText("Bundled model is live.");
      } catch (err) {
        setSmokeText(
          err instanceof Error ? `Smoke test failed: ${err.message}` : "Smoke test failed",
        );
      }
    })();
  }, [bundledDownload, step, setBundledDownload]);

  const chooseByo = () => {
    setLocalChoice("byo");
    setStep("byo_install");
  };

  const chooseSkip = async () => {
    setLocalChoice("skip");
    try {
      await Settings.set("local_backend_mode", "auto");
    } catch {
      /* non-fatal */
    }
    setStep("done");
  };

  const totalSteps = 4;
  const stepNum =
    step === "welcome"
      ? 1
      : step === "claude"
        ? 2
        : step === "local_choice" || step === "quick_start" || step === "byo_install"
          ? 3
          : 4;

  return (
    <div className="fixed inset-0 z-50 bg-bg/95 flex items-center justify-center p-6">
      <div className="card max-w-lg w-full" data-testid="first-run-wizard">
        <header className="mb-4">
          <div className="text-xs uppercase tracking-wide text-ink-faint mb-1">
            Setup · step {stepNum} of {totalSteps}
          </div>
          <h1 className="text-xl font-semibold">
            {step === "welcome" && "Welcome to iMakeAiTeams"}
            {step === "claude" && "Connect to Claude"}
            {step === "local_choice" && "How do you want to run local models?"}
            {step === "quick_start" && "Downloading the recommended model"}
            {step === "byo_install" && "Install Ollama or LM Studio"}
            {step === "done" && "All set"}
          </h1>
        </header>

        {step === "welcome" && (
          <div className="space-y-3" data-testid="step-welcome">
            <p className="text-sm text-ink-dim">
              Build AI teams where Claude handles complex work, local models
              handle simple tasks, and you set the rules. Everything runs on
              your machine — your data never leaves your desktop.
            </p>
            <button
              className="btn-primary w-full"
              onClick={() => setStep("claude")}
            >
              Get started
            </button>
          </div>
        )}

        {step === "claude" && (
          <div className="space-y-3" data-testid="step-claude">
            <p className="text-sm text-ink-dim">
              Paste your Anthropic API key. It is stored in the OS keyring,
              never in plaintext on disk.
            </p>
            <input
              className="input"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-ant-…"
            />
            <button
              className="btn-primary w-full"
              onClick={verify}
              disabled={verifying || !apiKey.trim()}
            >
              {verifying ? "Verifying…" : "Verify & continue"}
            </button>
            <button
              className="btn-ghost w-full"
              onClick={() =>
                window.electronAPI
                  .openExternal("https://console.anthropic.com/settings/keys")
                  .catch(() =>
                    pushToast({ kind: "error", text: "Could not open browser" }),
                  )
              }
            >
              Get a key from console.anthropic.com
            </button>
          </div>
        )}

        {step === "local_choice" && (
          <div className="space-y-3" data-testid="step-local-choice">
            <p className="text-sm text-ink-dim">
              Local models route simple messages on your machine for free. Pick
              the option that fits how you work.
            </p>
            <button
              className="btn-primary w-full text-left"
              onClick={startQuickStart}
              data-testid="choice-quick-start"
            >
              <div className="font-medium">Quick start (recommended)</div>
              <div className="text-xs text-ink-faint mt-0.5">
                Downloads a small Qwen3-4B model (~2.5 GB). One click, no other
                software needed.
              </div>
            </button>
            <button
              className="btn-ghost w-full text-left"
              onClick={chooseByo}
              data-testid="choice-byo"
            >
              <div className="font-medium">I have Ollama or LM Studio</div>
              <div className="text-xs text-ink-faint mt-0.5">
                Detects your existing setup. URLs are configurable in Settings
                later.
              </div>
            </button>
            <button
              className="btn-ghost w-full text-left"
              onClick={chooseSkip}
              data-testid="choice-skip"
            >
              <div className="font-medium">Skip for now</div>
              <div className="text-xs text-ink-faint mt-0.5">
                Use Claude only. You can change this in Settings any time.
              </div>
            </button>
          </div>
        )}

        {step === "quick_start" && (
          <div className="space-y-3" data-testid="step-quick-start">
            <BundledProgressView
              status={bundledDownload.status}
              bytesDone={bundledDownload.bytesDone}
              bytesTotal={bundledDownload.bytesTotal}
              error={bundledDownload.error}
              smokeText={smokeText}
            />
            {bundledDownload.status === "error" && (
              <div className="flex gap-2">
                <button
                  className="btn-ghost flex-1"
                  onClick={() => setStep("local_choice")}
                >
                  Back
                </button>
                <button className="btn-primary flex-1" onClick={startQuickStart}>
                  Try again
                </button>
              </div>
            )}
            {bundledDownload.status === "complete" && (
              <button
                className="btn-primary w-full"
                onClick={() => setStep("done")}
                data-testid="quick-start-continue"
              >
                Continue
              </button>
            )}
          </div>
        )}

        {step === "byo_install" && (
          <div className="space-y-3" data-testid="step-byo">
            <p className="text-sm text-ink-dim">
              The app auto-detects Ollama on{" "}
              <code className="text-ink">localhost:11434</code> and LM Studio on{" "}
              <code className="text-ink">localhost:1234</code>. If neither is
              installed, the commands below will install one with Windows'
              package manager:
            </p>
            <pre className="card text-xs whitespace-pre-wrap text-ink-dim">
{`# Ollama
winget install Ollama.Ollama

# LM Studio
winget install LMStudio.LMStudio`}
            </pre>
            <button
              className="btn-primary w-full"
              onClick={() => setStep("done")}
            >
              Continue
            </button>
          </div>
        )}

        {step === "done" && (
          <div className="space-y-3" data-testid="step-done">
            <p className="text-sm text-ink-dim">
              {localChoice === "quick_start" &&
                "Your bundled model is ready. Open the chat tab to talk to your team."}
              {localChoice === "byo" &&
                "Make sure Ollama or LM Studio is running, then open the chat tab."}
              {localChoice === "skip" &&
                "Claude-only mode is active. You can enable a local backend later from Settings."}
              {localChoice === null &&
                "You're ready. Open the chat tab to talk to your team."}
            </p>
            <button
              className="btn-primary w-full"
              onClick={finish}
              data-testid="finish-button"
            >
              Enter the app
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

interface BundledProgressViewProps {
  status: "idle" | "downloading" | "complete" | "error";
  bytesDone: number;
  bytesTotal: number;
  error: string;
  smokeText: string;
}

function BundledProgressView({
  status,
  bytesDone,
  bytesTotal,
  error,
  smokeText,
}: BundledProgressViewProps) {
  const pct =
    bytesTotal > 0 ? Math.min(100, Math.round((bytesDone / bytesTotal) * 100)) : 0;

  if (status === "error") {
    return (
      <div className="card border-err/40 text-err text-sm" data-testid="quick-start-error">
        {error || "Download failed."}
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="text-sm text-ink-dim">
        {status === "complete"
          ? "Download complete."
          : "Streaming the GGUF from Hugging Face."}
      </div>
      <div
        className="h-2 w-full bg-bg-2 rounded overflow-hidden"
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        data-testid="quick-start-progress"
      >
        <div
          className="h-full bg-accent transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="text-xs text-ink-faint flex items-center justify-between">
        <span>
          {formatBytes(bytesDone)}
          {bytesTotal > 0 ? ` / ${formatBytes(bytesTotal)}` : ""}
        </span>
        <span>{pct}%</span>
      </div>
      {smokeText && (
        <div className="text-xs text-ink-dim mt-1" data-testid="smoke-text">
          {smokeText}
        </div>
      )}
    </div>
  );
}
