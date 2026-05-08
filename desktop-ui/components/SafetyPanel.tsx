import { useCallback, useEffect, useState } from "react";

import { Safety } from "@/api/client";
import { t } from "@/i18n";
import { useAppStore } from "@/stores/appStore";
import type {
  SafetyDenialReason,
  SafetyMastEntry,
  SafetySummary,
} from "@/types/safety";

const WINDOW_OPTIONS: number[] = [7, 30, 90];

// Best-effort translation of backend-provided strings to user-facing labels.
// Keys in en.json are normalised (lowercase, snake_case); backend strings can
// be free-form (e.g. "tool not in allowlist"). Slugify and prefix; fall back
// to the raw text when no key matches so new categories surface automatically
// instead of silently disappearing.
function slugify(s: string): string {
  return s.toLowerCase().trim().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
}

function reasonLabel(raw: string): string {
  const key = `governance.${slugify(raw)}`;
  const translated = t(key);
  return translated === key ? raw : translated;
}

function mastLabel(raw: string): string {
  const key = `mast.${slugify(raw)}`;
  const translated = t(key);
  return translated === key ? raw : translated;
}

function formatTimestamp(iso: string | null): string {
  if (!iso) return "—";
  return iso.slice(0, 19).replace("T", " ");
}

interface CardProps {
  testId: string;
  title: string;
  metric: number;
  metricLabel: string;
  empty: boolean;
  emptyText: string;
  children?: React.ReactNode;
}

function Card({
  testId,
  title,
  metric,
  metricLabel,
  empty,
  emptyText,
  children,
}: CardProps) {
  return (
    <div className="card" data-testid={testId}>
      <div className="text-xs uppercase tracking-wide text-ink-faint">{title}</div>
      <div className="mt-1 flex items-baseline gap-2">
        <div className="text-3xl font-semibold tabular-nums">{metric}</div>
        <div className="text-xs text-ink-dim">{metricLabel}</div>
      </div>
      <div className="mt-3 text-sm text-ink-dim">
        {empty ? (
          <span data-testid={`${testId}-empty`} className="text-ink-faint">
            {emptyText}
          </span>
        ) : (
          children
        )}
      </div>
    </div>
  );
}

export function SafetyPanel() {
  const ready = useAppStore((s) => s.sidecarStatus?.status === "ready");
  const pushToast = useAppStore((s) => s.pushToast);
  const [days, setDays] = useState<number>(30);
  const [summary, setSummary] = useState<SafetySummary | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const fetchSummary = useCallback(
    async (windowDays: number) => {
      setLoading(true);
      try {
        const data = await Safety.getSafetySummary(windowDays);
        setSummary(data);
      } catch (err) {
        pushToast({
          kind: "error",
          text: err instanceof Error ? err.message : "Failed to load safety summary",
        });
      } finally {
        setLoading(false);
      }
    },
    [pushToast],
  );

  useEffect(() => {
    if (!ready) return;
    void fetchSummary(days);
  }, [ready, days, fetchSummary]);

  const esc        = summary?.escalations;
  const memory     = summary?.memory_gate;
  const canary     = summary?.canary;
  const governance = summary?.governance;
  const routing    = summary?.routing;
  const voting     = summary?.voting;

  return (
    <div className="p-6 overflow-y-auto h-full" data-testid="safety-panel">
      <header className="mb-4">
        <h1 className="text-xl font-semibold">{t("safety.title")}</h1>
        <p className="text-sm text-ink-dim">{t("safety.subtitle")}</p>
      </header>

      <div className="mb-4 flex items-center gap-2">
        <div className="flex gap-1" role="group" aria-label="Time window">
          {WINDOW_OPTIONS.map((w) => (
            <button
              key={w}
              type="button"
              onClick={() => setDays(w)}
              data-testid={`safety-window-${w}`}
              aria-pressed={days === w}
              className={`px-3 py-1 text-sm rounded-md border transition ${
                days === w
                  ? "border-accent bg-accent/10 text-ink"
                  : "border-line text-ink-dim hover:bg-bg-2"
              }`}
            >
              {t(`safety.window.${w}`)}
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={() => void fetchSummary(days)}
          data-testid="safety-refresh"
          disabled={!ready || loading}
          className="px-3 py-1 text-sm rounded-md border border-line text-ink-dim hover:bg-bg-2 disabled:opacity-50"
        >
          {t("safety.refresh")}
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {/* ── Escalations ──────────────────────────────────────────────── */}
        <Card
          testId="safety-card-escalations"
          title={t("safety.escalations.title")}
          metric={esc?.triggered ?? 0}
          metricLabel={t("safety.escalations.metric")}
          empty={!esc || esc.triggered === 0}
          emptyText={t("safety.escalations.empty")}
        >
          <ul className="space-y-1">
            <li className="flex justify-between">
              <span>{t("safety.escalations.approved")}</span>
              <span className="tabular-nums">{esc?.approved ?? 0}</span>
            </li>
            <li className="flex justify-between">
              <span>{t("safety.escalations.denied")}</span>
              <span className="tabular-nums">{esc?.denied ?? 0}</span>
            </li>
            <li className="flex justify-between">
              <span>{t("safety.escalations.pending")}</span>
              <span className="tabular-nums">{esc?.pending ?? 0}</span>
            </li>
          </ul>
        </Card>

        {/* ── Memory Gate ──────────────────────────────────────────────── */}
        <Card
          testId="safety-card-memory"
          title={t("safety.memory.title")}
          metric={memory?.facts_proposed ?? 0}
          metricLabel={t("safety.memory.metric")}
          empty={!memory || memory.facts_proposed === 0}
          emptyText={t("safety.memory.empty")}
        >
          <ul className="space-y-1">
            <li className="flex justify-between">
              <span>{t("safety.memory.auto_accepted")}</span>
              <span className="tabular-nums">{memory?.auto_accepted ?? 0}</span>
            </li>
            <li className="flex justify-between">
              <span>{t("safety.memory.user_approved")}</span>
              <span className="tabular-nums">{memory?.user_approved ?? 0}</span>
            </li>
            <li className="flex justify-between">
              <span>{t("safety.memory.user_denied")}</span>
              <span className="tabular-nums">{memory?.user_denied ?? 0}</span>
            </li>
            <li className="flex justify-between">
              <span>{t("safety.memory.pending")}</span>
              <span className="tabular-nums">{memory?.pending ?? 0}</span>
            </li>
          </ul>
        </Card>

        {/* ── Canary ───────────────────────────────────────────────────── */}
        <Card
          testId="safety-card-canary"
          title={t("safety.canary.title")}
          metric={canary?.alerts_fired ?? 0}
          metricLabel={t("safety.canary.metric")}
          empty={!canary || canary.alerts_fired === 0}
          emptyText={t("safety.canary.empty")}
        >
          <ul className="space-y-1">
            <li className="flex justify-between">
              <span>{t("safety.canary.baselines")}</span>
              <span className="tabular-nums">{canary?.baselines ?? 0}</span>
            </li>
            <li className="flex justify-between">
              <span>{t("safety.canary.last_alert")}</span>
              <span className="tabular-nums">
                {formatTimestamp(canary?.last_alert_at ?? null)}
              </span>
            </li>
          </ul>
        </Card>

        {/* ── Governance ───────────────────────────────────────────────── */}
        <Card
          testId="safety-card-governance"
          title={t("safety.governance.title")}
          metric={governance?.tool_calls_denied ?? 0}
          metricLabel={t("safety.governance.metric")}
          empty={!governance || governance.tool_calls_denied === 0}
          emptyText={t("safety.governance.empty")}
        >
          <div className="text-[11px] uppercase tracking-wide text-ink-faint mb-1">
            {t("safety.governance.top_reasons")}
          </div>
          <ul className="space-y-1">
            {(governance?.denial_top_reasons ?? [])
              .slice(0, 3)
              .map((r: SafetyDenialReason) => (
                <li key={r.reason} className="flex justify-between gap-2">
                  <span className="truncate">{reasonLabel(r.reason)}</span>
                  <span className="tabular-nums shrink-0">{r.count}</span>
                </li>
              ))}
          </ul>
        </Card>

        {/* ── Routing ──────────────────────────────────────────────────── */}
        <Card
          testId="safety-card-routing"
          title={t("safety.routing.title")}
          metric={routing?.turns_failed ?? 0}
          metricLabel={t("safety.routing.metric")}
          empty={!routing || routing.turns_failed === 0}
          emptyText={t("safety.routing.empty")}
        >
          <div className="flex justify-between mb-2">
            <span>{t("safety.routing.total")}</span>
            <span className="tabular-nums">{routing?.turns_total ?? 0}</span>
          </div>
          <div className="text-[11px] uppercase tracking-wide text-ink-faint mb-1">
            {t("safety.routing.mast_breakdown")}
          </div>
          <ul className="space-y-1">
            {(routing?.mast_breakdown ?? []).map((m: SafetyMastEntry) => (
              <li key={m.category} className="flex justify-between">
                <span>{mastLabel(m.category)}</span>
                <span className="tabular-nums">{m.count}</span>
              </li>
            ))}
          </ul>
        </Card>

        {/* ── Voting ───────────────────────────────────────────────────── */}
        <Card
          testId="safety-card-voting"
          title={t("safety.voting.title")}
          metric={voting?.high_stakes_turns ?? 0}
          metricLabel={t("safety.voting.metric")}
          empty={!voting || voting.high_stakes_turns === 0}
          emptyText={t("safety.voting.empty")}
        >
          <ul className="space-y-1">
            <li className="flex justify-between">
              <span>{t("safety.voting.consensus")}</span>
              <span className="tabular-nums">
                {voting?.consensus_reached ?? 0}
                {voting && voting.high_stakes_turns > 0
                  ? ` / ${voting.high_stakes_turns}`
                  : ""}
              </span>
            </li>
          </ul>
        </Card>
      </div>
    </div>
  );
}
