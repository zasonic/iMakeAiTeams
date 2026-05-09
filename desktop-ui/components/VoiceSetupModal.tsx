// desktop-ui/components/VoiceSetupModal.tsx — first-time-voice download
// modal (PR 17). Mirrors FirstRunWizard's quick-start path: shows download
// size, an SSE-driven progress bar, and completes when both STT + TTS
// assets are ready.
//
// The modal is fired by SettingsPanel when the user toggles voice on for
// the first time AND the assets aren't already on disk. We don't auto-open
// on app launch — voice is opt-in.

import { useEffect, useState } from "react";

import { Voice } from "@/api/client";
import { useAppStore, type VoiceAssetsState } from "@/stores/appStore";

interface Props {
  open: boolean;
  onClose: () => void;
  onComplete: () => void;
}

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

function _percent(done: number, total: number): number {
  if (total <= 0) return 0;
  return Math.min(100, Math.round((done / total) * 100));
}

export function VoiceSetupModal({ open, onClose, onComplete }: Props) {
  const pushToast = useAppStore((s) => s.pushToast);
  const voiceAssets = useAppStore((s) => s.voiceAssets);
  const setVoiceAssets = useAppStore((s) => s.setVoiceAssets);
  const [starting, setStarting] = useState(false);

  const startDownload = async () => {
    setStarting(true);
    setVoiceAssets({
      status: "downloading",
      sttReady: false,
      ttsReady: false,
      sttBytesDone: 0,
      sttBytesTotal: 0,
      ttsBytesDone: 0,
      ttsBytesTotal: 0,
      error: "",
    });
    try {
      const r = await Voice.assetsDownload();
      if (!r.ok) {
        setVoiceAssets({
          status: "error",
          sttReady: false,
          ttsReady: false,
          sttBytesDone: 0,
          sttBytesTotal: 0,
          ttsBytesDone: 0,
          ttsBytesTotal: 0,
          error: r.error ?? "Download could not start",
        });
        pushToast({ kind: "error", text: r.error ?? "Download could not start" });
      }
    } catch (err) {
      setVoiceAssets({
        status: "error",
        sttReady: false,
        ttsReady: false,
        sttBytesDone: 0,
        sttBytesTotal: 0,
        ttsBytesDone: 0,
        ttsBytesTotal: 0,
        error: err instanceof Error ? err.message : "Download could not start",
      });
    } finally {
      setStarting(false);
    }
  };

  // Auto-fire onComplete when both assets land. Guarded by ``open`` so a
  // background completion (the user already closed the modal) doesn't
  // trigger a stale onComplete that re-opens panels.
  useEffect(() => {
    if (!open) return;
    if (voiceAssets.status === "complete" && voiceAssets.sttReady && voiceAssets.ttsReady) {
      onComplete();
    }
  }, [open, voiceAssets.status, voiceAssets.sttReady, voiceAssets.ttsReady, onComplete]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 bg-bg/95 flex items-center justify-center p-6"
      data-testid="voice-setup-modal"
    >
      <div className="card max-w-lg w-full">
        <header className="mb-4">
          <h1 className="text-xl font-semibold">Set up voice</h1>
          <p className="text-sm text-ink-dim mt-1">
            Voice input transcribes what you say with Whisper.cpp; voice output
            reads assistant replies aloud with Piper. The first download is
            roughly <strong>200&nbsp;MB</strong>; everything runs on your
            machine after that — no audio is ever sent off your computer.
          </p>
        </header>

        <ProgressView state={voiceAssets} />

        {voiceAssets.status === "idle" && (
          <div className="flex gap-2 mt-4">
            <button
              type="button"
              className="btn-ghost flex-1"
              onClick={onClose}
              data-testid="voice-setup-cancel"
            >
              Not now
            </button>
            <button
              type="button"
              className="btn-primary flex-1"
              onClick={startDownload}
              disabled={starting}
              data-testid="voice-setup-start"
            >
              {starting ? "Starting…" : "Download voice models"}
            </button>
          </div>
        )}

        {voiceAssets.status === "error" && (
          <div className="flex gap-2 mt-4">
            <button
              type="button"
              className="btn-ghost flex-1"
              onClick={onClose}
            >
              Close
            </button>
            <button
              type="button"
              className="btn-primary flex-1"
              onClick={startDownload}
              data-testid="voice-setup-retry"
            >
              Try again
            </button>
          </div>
        )}

        {voiceAssets.status === "complete" && (
          <button
            type="button"
            className="btn-primary w-full mt-4"
            onClick={onComplete}
            data-testid="voice-setup-done"
          >
            Done
          </button>
        )}
      </div>
    </div>
  );
}

function ProgressView({ state }: { state: VoiceAssetsState }) {
  if (state.status === "idle") {
    return (
      <ul className="text-xs text-ink-dim space-y-1 mt-2">
        <li>• Speech-to-text model (Whisper base.en, ~150 MB)</li>
        <li>• Text-to-speech voice (Piper Amy, ~60 MB)</li>
      </ul>
    );
  }
  if (state.status === "error") {
    return (
      <div
        className="card border-err/40 text-err text-sm"
        data-testid="voice-setup-error"
      >
        {state.error || "Download failed."}
      </div>
    );
  }

  const sttPct = _percent(state.sttBytesDone, state.sttBytesTotal);
  const ttsPct = _percent(state.ttsBytesDone, state.ttsBytesTotal);
  const overall = Math.round((sttPct + ttsPct) / 2);

  return (
    <div className="space-y-3" data-testid="voice-setup-progress">
      <div>
        <div className="flex items-center justify-between text-xs text-ink-dim mb-1">
          <span>Speech-to-text {state.sttReady && "✓"}</span>
          <span>{sttPct}%</span>
        </div>
        <div
          className="h-2 w-full bg-bg-2 rounded overflow-hidden"
          role="progressbar"
          aria-label="STT download progress"
          aria-valuenow={sttPct}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <div
            className="h-full bg-accent transition-all"
            style={{ width: `${sttPct}%` }}
          />
        </div>
        <div className="text-[11px] text-ink-faint mt-0.5">
          {formatBytes(state.sttBytesDone)}
          {state.sttBytesTotal > 0 ? ` / ${formatBytes(state.sttBytesTotal)}` : ""}
        </div>
      </div>

      <div>
        <div className="flex items-center justify-between text-xs text-ink-dim mb-1">
          <span>Text-to-speech {state.ttsReady && "✓"}</span>
          <span>{ttsPct}%</span>
        </div>
        <div
          className="h-2 w-full bg-bg-2 rounded overflow-hidden"
          role="progressbar"
          aria-label="TTS download progress"
          aria-valuenow={ttsPct}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <div
            className="h-full bg-accent transition-all"
            style={{ width: `${ttsPct}%` }}
          />
        </div>
        <div className="text-[11px] text-ink-faint mt-0.5">
          {formatBytes(state.ttsBytesDone)}
          {state.ttsBytesTotal > 0 ? ` / ${formatBytes(state.ttsBytesTotal)}` : ""}
        </div>
      </div>

      <div className="text-xs text-ink-dim">
        {state.status === "complete"
          ? "Done."
          : `Overall progress: ${overall}%`}
      </div>
    </div>
  );
}
