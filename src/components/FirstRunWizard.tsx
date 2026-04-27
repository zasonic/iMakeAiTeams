import { useState } from "react";

import { Settings } from "@/api/client";
import { useAppStore } from "@/stores/appStore";

interface Props {
  onComplete: () => void;
}

export function FirstRunWizard({ onComplete }: Props) {
  const pushToast = useAppStore((s) => s.pushToast);
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [apiKey, setApiKey] = useState("");
  const [verifying, setVerifying] = useState(false);

  const verify = async () => {
    if (!apiKey.trim()) return;
    setVerifying(true);
    try {
      const rsp = await Settings.verifyApiKey(apiKey);
      if (rsp.ok) {
        pushToast({ kind: "success", text: rsp.message });
        setStep(2);
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
            Setup · step {step} of 3
          </div>
          <h1 className="text-xl font-semibold">
            {step === 1 && "Connect to Claude"}
            {step === 2 && "Local models"}
            {step === 3 && "All set"}
          </h1>
        </header>

        {step === 1 && (
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

        {step === 2 && (
          <div className="space-y-3">
            <p className="text-sm text-ink-dim">
              Optional: install Ollama or LM Studio to route simple messages
              through a local model. You can configure URLs later in Settings.
            </p>
            <button
              className="btn-primary w-full"
              onClick={() => setStep(3)}
            >
              Continue
            </button>
          </div>
        )}

        {step === 3 && (
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
