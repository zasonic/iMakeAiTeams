import { useState } from "react";

import { Settings } from "@/api/client";
import { useAppStore } from "@/stores/appStore";

interface Props {
  onComplete: () => void;
}

export function FirstRunWizard({ onComplete }: Props) {
  const pushToast = useAppStore((s) => s.pushToast);
  const [step, setStep] = useState<1 | 2 | 3 | 4 | 5 | 6>(1);
  const [apiKey, setApiKey] = useState("");
  const [verifying, setVerifying] = useState(false);

  const verify = async () => {
    if (!apiKey.trim()) return;
    setVerifying(true);
    try {
      const rsp = await Settings.verifyApiKey(apiKey);
      if (rsp.ok) {
        pushToast({ kind: "success", text: rsp.message });
        setStep(3);
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

  return (
    <div className="fixed inset-0 z-50 bg-bg/95 flex items-center justify-center p-6">
      <div className="card max-w-lg w-full">
        <header className="mb-4">
          <div className="text-xs uppercase tracking-wide text-ink-faint mb-1">
            Setup · step {step} of 6
          </div>
          <h1 className="text-xl font-semibold">
            {step === 1 && "Welcome to iMakeAiTeams"}
            {step === 2 && "Connect to Claude"}
            {step === 3 && "Local models"}
            {step === 4 && "Your workspace"}
            {step === 5 && "When you're ready for more"}
            {step === 6 && "All set"}
          </h1>
        </header>

        {step === 1 && (
          <div className="space-y-3">
            <p className="text-sm text-ink-dim">
              Build AI teams where Claude handles complex work, local models
              handle simple tasks, and you set the rules. Everything runs on
              your machine — your data never leaves your desktop.
            </p>
            <button
              className="btn-primary w-full"
              onClick={() => setStep(2)}
            >
              Get started
            </button>
          </div>
        )}

        {step === 2 && (
          <div className="space-y-3">
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

        {step === 3 && (
          <div className="space-y-3">
            <p className="text-sm text-ink-dim">
              For best results, install a local AI model runner. This lets
              simple messages route locally (free) instead of going to Claude.
            </p>
            <p className="text-sm text-ink-dim">
              <strong>Ollama</strong> is recommended — it is free, lightweight,
              and works out of the box.
            </p>
            <button
              className="btn-ghost w-full"
              onClick={() =>
                window.electronAPI
                  .openExternal("https://ollama.com/download")
                  .catch(() =>
                    pushToast({ kind: "error", text: "Could not open browser" }),
                  )
              }
            >
              Download Ollama (free)
            </button>
            <p className="text-xs text-ink-faint">
              Already have Ollama or LM Studio? The app detects them
              automatically. You can configure URLs in Settings later.
            </p>
            <button
              className="btn-primary w-full"
              onClick={() => setStep(4)}
            >
              Continue
            </button>
          </div>
        )}

        {step === 4 && (
          <div className="space-y-3">
            <p className="text-sm text-ink-dim">
              <strong>Chat</strong> — Talk to your AI team. Messages are routed
              to the best model automatically.
            </p>
            <p className="text-sm text-ink-dim">
              <strong>Agents</strong> — Create specialist agents with custom
              instructions and personalities.
            </p>
            <p className="text-sm text-ink-dim">
              <strong>Documents</strong> — Add files and folders so your team
              can reference them in conversation.
            </p>
            <p className="text-sm text-ink-dim">
              <strong>Memory</strong> — Your team remembers facts across
              conversations automatically.
            </p>
            <p className="text-sm text-ink-dim">
              <strong>Settings</strong> — Manage API keys, choose models, and
              control routing behavior.
            </p>
            <button
              className="btn-primary w-full"
              onClick={() => setStep(5)}
            >
              Continue
            </button>
          </div>
        )}

        {step === 5 && (
          <div className="space-y-3">
            <p className="text-sm text-ink-dim">
              At the bottom of the sidebar there's a Studio Mode toggle. Turn
              it on to access advanced features: prompt engineering, MCP tool
              servers, security scanning, and diagnostics. You don't need
              these to get started.
            </p>
            <button
              className="btn-primary w-full"
              onClick={() => setStep(6)}
            >
              Continue
            </button>
          </div>
        )}

        {step === 6 && (
          <div className="space-y-3">
            <p className="text-sm text-ink-dim">
              You're ready. Open the chat tab to talk to your team.
            </p>
            <button className="btn-primary w-full" onClick={finish}>
              Enter the app
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
