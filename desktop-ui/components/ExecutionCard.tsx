// desktop-ui/components/ExecutionCard.tsx — single Power Mode execution step.
//
// Renders one OpenClaw event (planning, tool call, file write, shell command,
// web action) as a collapsible card. Default-expanded while running, collapses
// once status flips to "done" or "error" so the chat doesn't bloat.

import { useState } from "react";

import type { ExecutionStep } from "@/stores/appStore";

interface ExecutionCardProps {
  step: ExecutionStep;
}

const KIND_LABEL: Record<ExecutionStep["kind"], string> = {
  thinking: "Planning",
  tool_call: "Tool",
  file_write: "File",
  shell: "Shell",
  web: "Web",
  other: "Step",
};

const KIND_ICON: Record<ExecutionStep["kind"], string> = {
  thinking: "·",
  tool_call: "*",
  file_write: "+",
  shell: ">",
  web: "@",
  other: "-",
};

function statusTone(status: ExecutionStep["status"]): string {
  if (status === "running") return "border-warn/40 bg-warn/5 text-ink";
  if (status === "error") return "border-err/40 bg-err/5 text-err";
  return "border-line bg-bg-2 text-ink";
}

function copyToClipboard(text: string): void {
  navigator.clipboard?.writeText(text).catch(() => {});
}

function summaryFor(step: ExecutionStep): string {
  switch (step.kind) {
    case "thinking":
      return step.title || "Planning";
    case "tool_call":
      return step.title || "Tool call";
    case "file_write":
      return step.path || "File";
    case "shell":
      return step.command || "Shell command";
    case "web":
      return step.title || step.url || "Web";
    default:
      return step.title || "Step";
  }
}

export function ExecutionCard({ step }: ExecutionCardProps) {
  const [expanded, setExpanded] = useState<boolean>(step.status === "running");
  const tone = statusTone(step.status);

  return (
    <div className={`rounded-md border ${tone} text-xs overflow-hidden`}>
      <button
        type="button"
        onClick={() => setExpanded((e) => !e)}
        className="w-full text-left px-3 py-1.5 flex items-center gap-2 hover:bg-bg-1/40"
      >
        <span aria-hidden className="font-mono text-ink-faint w-4 text-center">
          {KIND_ICON[step.kind]}
        </span>
        <span className="font-medium text-ink-dim w-14 flex-shrink-0">
          {KIND_LABEL[step.kind]}
        </span>
        <span className="truncate flex-1">{summaryFor(step)}</span>
        <span className="text-[10px] uppercase tracking-wide text-ink-faint">
          {step.status === "running" ? "running…" : step.status}
        </span>
      </button>
      {expanded && (
        <div className="border-t border-line/60 px-3 py-2 space-y-1.5 font-mono text-[11px] leading-relaxed">
          {step.kind === "thinking" && step.detail && (
            <pre className="whitespace-pre-wrap text-ink-dim">{step.detail}</pre>
          )}
          {step.kind === "tool_call" && (
            <>
              {step.args !== undefined && (
                <DetailBlock label="args" value={JSON.stringify(step.args, null, 2)} />
              )}
              {step.result !== undefined && (
                <DetailBlock
                  label="result"
                  value={typeof step.result === "string"
                    ? step.result
                    : JSON.stringify(step.result, null, 2)}
                />
              )}
            </>
          )}
          {step.kind === "file_write" && (
            <>
              {step.path && <DetailBlock label="path" value={step.path} />}
              {step.preview && <DetailBlock label="preview" value={step.preview} />}
              {typeof step.bytes === "number" && (
                <div className="text-ink-faint">{step.bytes} bytes</div>
              )}
            </>
          )}
          {step.kind === "shell" && (
            <>
              {step.command && <DetailBlock label="$" value={step.command} />}
              {step.stdout && <DetailBlock label="stdout" value={step.stdout} />}
              {step.stderr && <DetailBlock label="stderr" value={step.stderr} />}
              {typeof step.exit_code === "number" && (
                <div className="text-ink-faint">exit {step.exit_code}</div>
              )}
            </>
          )}
          {step.kind === "web" && (
            <>
              {step.url && <DetailBlock label="url" value={step.url} />}
              {step.summary && <DetailBlock label="summary" value={step.summary} />}
            </>
          )}
          {step.kind === "other" && step.detail && (
            <pre className="whitespace-pre-wrap text-ink-dim">{step.detail}</pre>
          )}
        </div>
      )}
    </div>
  );
}

interface DetailBlockProps {
  label: string;
  value: string;
}

function DetailBlock({ label, value }: DetailBlockProps) {
  return (
    <div className="space-y-0.5">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-ink-faint">
        <span>{label}</span>
        <button
          type="button"
          className="hover:text-ink"
          onClick={() => copyToClipboard(value)}
          title="Copy to clipboard"
        >
          copy
        </button>
      </div>
      <pre className="whitespace-pre-wrap break-words text-ink">{value}</pre>
    </div>
  );
}

