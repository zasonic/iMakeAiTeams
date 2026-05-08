import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Usage } from "@/api/client";
import { t } from "@/i18n";
import { useAppStore } from "@/stores/appStore";
import type {
  UsageByAgent,
  UsageByModel,
  UsageGroupBy,
  UsageRow,
  UsageSummary,
} from "@/types/usage";

const WINDOW_OPTIONS: number[] = [7, 30, 90];
const GROUP_BY_OPTIONS: UsageGroupBy[] = ["day", "model", "agent"];

// Mirrors tailwind.config.js → theme.extend.colors. Recharts can't read
// Tailwind classes, so the hexes are duplicated here and must move in
// lock-step with the config.
const COLOR_ACCENT  = "#8b7cf6";       // colors.accent.DEFAULT
const COLOR_LINE    = "#2a2a33";       // colors.line.DEFAULT
const COLOR_INK_DIM = "#9d9db0";       // colors.ink.dim

function fmtUsd(value: number): string {
  return value.toLocaleString(undefined, {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 4,
  });
}

function fmtInt(value: number): string {
  return value.toLocaleString();
}

interface ChartDatum {
  key:           string;
  cost_usd:      number;
  input_tokens:  number;
  output_tokens: number;
  turns:         number;
}

interface ChartTooltipProps {
  active?: boolean;
  payload?: { payload: ChartDatum }[];
  label?: string;
}

function ChartTooltip({ active, payload }: ChartTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const d = payload[0].payload;
  return (
    <div
      className="rounded-md border border-line bg-bg-1 px-3 py-2 text-xs shadow-glass"
      data-testid="usage-chart-tooltip"
    >
      <div className="font-semibold text-ink mb-1 truncate max-w-[16rem]">{d.key}</div>
      <div className="flex justify-between gap-4 text-ink-dim">
        <span>{t("usage.tooltip.cost")}</span>
        <span className="tabular-nums text-ink">{fmtUsd(d.cost_usd)}</span>
      </div>
      <div className="flex justify-between gap-4 text-ink-dim">
        <span>{t("usage.tooltip.input")}</span>
        <span className="tabular-nums">{fmtInt(d.input_tokens)}</span>
      </div>
      <div className="flex justify-between gap-4 text-ink-dim">
        <span>{t("usage.tooltip.output")}</span>
        <span className="tabular-nums">{fmtInt(d.output_tokens)}</span>
      </div>
    </div>
  );
}

interface StatCardProps {
  testId: string;
  label:  string;
  value:  string;
}

function StatCard({ testId, label, value }: StatCardProps) {
  return (
    <div className="card" data-testid={testId}>
      <div className="text-xs uppercase tracking-wide text-ink-faint">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums">{value}</div>
    </div>
  );
}

function rowKeyLabel(row: UsageRow, groupBy: UsageGroupBy): string {
  if (row.key !== "") return row.key;
  return groupBy === "agent" ? t("usage.no_agent") : "—";
}

export function UsagePanel() {
  const ready = useAppStore((s) => s.sidecarStatus?.status === "ready");
  const pushToast = useAppStore((s) => s.pushToast);
  const [days, setDays] = useState<number>(30);
  const [groupBy, setGroupBy] = useState<UsageGroupBy>("day");
  const [summary, setSummary] = useState<UsageSummary | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const fetchSummary = useCallback(
    async (windowDays: number, groupBy: UsageGroupBy) => {
      setLoading(true);
      try {
        const data = await Usage.getUsageSummary(windowDays, groupBy);
        setSummary(data);
      } catch (err) {
        pushToast({
          kind: "error",
          text: err instanceof Error ? err.message : "Failed to load usage summary",
        });
      } finally {
        setLoading(false);
      }
    },
    [pushToast],
  );

  useEffect(() => {
    if (!ready) return;
    void fetchSummary(days, groupBy);
  }, [ready, days, groupBy, fetchSummary]);

  const chartData: ChartDatum[] = useMemo(() => {
    if (!summary) return [];
    return summary.rows.map((r) => ({
      key:           rowKeyLabel(r, summary.group_by),
      cost_usd:      r.cost_usd,
      input_tokens:  r.input_tokens,
      output_tokens: r.output_tokens,
      turns:         r.turns,
    }));
  }, [summary]);

  const total      = summary?.total;
  const topModels  = summary?.by_model ?? [];
  const topAgents  = summary?.by_agent ?? [];
  const isEmpty    = chartData.length === 0;

  return (
    <div className="p-6 overflow-y-auto h-full" data-testid="usage-panel">
      <header className="mb-4">
        <h1 className="text-xl font-semibold">{t("usage.title")}</h1>
        <p className="text-sm text-ink-dim">{t("usage.subtitle")}</p>
      </header>

      <div className="mb-4 flex items-center gap-2 flex-wrap">
        <div className="flex gap-1" role="group" aria-label="Time window">
          {WINDOW_OPTIONS.map((w) => (
            <button
              key={w}
              type="button"
              onClick={() => setDays(w)}
              data-testid={`usage-window-${w}`}
              aria-pressed={days === w}
              className={`px-3 py-1 text-sm rounded-md border transition ${
                days === w
                  ? "border-accent bg-accent/10 text-ink"
                  : "border-line text-ink-dim hover:bg-bg-2"
              }`}
            >
              {t(`usage.window.${w}`)}
            </button>
          ))}
        </div>
        <div className="flex gap-1" role="group" aria-label="Group by">
          {GROUP_BY_OPTIONS.map((g) => (
            <button
              key={g}
              type="button"
              onClick={() => setGroupBy(g)}
              data-testid={`usage-group-${g}`}
              aria-pressed={groupBy === g}
              className={`px-3 py-1 text-sm rounded-md border transition ${
                groupBy === g
                  ? "border-accent bg-accent/10 text-ink"
                  : "border-line text-ink-dim hover:bg-bg-2"
              }`}
            >
              {t(`usage.group_by.${g}`)}
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={() => void fetchSummary(days, groupBy)}
          data-testid="usage-refresh"
          disabled={!ready || loading}
          className="px-3 py-1 text-sm rounded-md border border-line text-ink-dim hover:bg-bg-2 disabled:opacity-50"
        >
          {t("usage.refresh")}
        </button>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
        <StatCard
          testId="usage-stat-cost"
          label={t("usage.total.cost")}
          value={fmtUsd(total?.cost_usd ?? 0)}
        />
        <StatCard
          testId="usage-stat-turns"
          label={t("usage.total.turns")}
          value={fmtInt(total?.turns ?? 0)}
        />
        <StatCard
          testId="usage-stat-input"
          label={t("usage.total.input")}
          value={fmtInt(total?.input_tokens ?? 0)}
        />
        <StatCard
          testId="usage-stat-output"
          label={t("usage.total.output")}
          value={fmtInt(total?.output_tokens ?? 0)}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <div className="card lg:col-span-2" data-testid="usage-chart-card">
          {isEmpty ? (
            <div
              className="h-72 flex items-center justify-center text-sm text-ink-faint"
              data-testid="usage-chart-empty"
            >
              {t("usage.empty")}
            </div>
          ) : (
            <div className="h-72" data-testid="usage-chart">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 24 }}>
                  <CartesianGrid stroke={COLOR_LINE} strokeDasharray="3 3" vertical={false} />
                  <XAxis
                    dataKey="key"
                    stroke={COLOR_INK_DIM}
                    fontSize={11}
                    tickLine={false}
                    interval="preserveStartEnd"
                    angle={-20}
                    textAnchor="end"
                    height={48}
                  />
                  <YAxis
                    stroke={COLOR_INK_DIM}
                    fontSize={11}
                    tickLine={false}
                    tickFormatter={(v: number) => fmtUsd(v)}
                  />
                  <Tooltip
                    cursor={{ fill: COLOR_LINE, fillOpacity: 0.3 }}
                    content={<ChartTooltip />}
                  />
                  <Bar
                    dataKey="cost_usd"
                    fill={COLOR_ACCENT}
                    radius={[4, 4, 0, 0]}
                    isAnimationActive={false}
                    data-testid="usage-chart-bars"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        <div className="flex flex-col gap-3">
          <div className="card" data-testid="usage-top-models">
            <div className="text-xs uppercase tracking-wide text-ink-faint mb-2">
              {t("usage.top_models")}
            </div>
            {topModels.length === 0 ? (
              <div
                className="text-sm text-ink-faint"
                data-testid="usage-top-models-empty"
              >
                {t("usage.empty")}
              </div>
            ) : (
              <table className="w-full text-sm">
                <thead className="text-[10px] uppercase tracking-wide text-ink-faint">
                  <tr>
                    <th className="text-left font-normal pb-1">{t("usage.col.model")}</th>
                    <th className="text-right font-normal pb-1">{t("usage.col.cost")}</th>
                    <th className="text-right font-normal pb-1">{t("usage.col.turns")}</th>
                  </tr>
                </thead>
                <tbody>
                  {topModels.map((m: UsageByModel) => (
                    <tr key={m.model} data-testid={`usage-top-model-${m.model}`}>
                      <td className="truncate pr-2">{m.model || "—"}</td>
                      <td className="text-right tabular-nums">{fmtUsd(m.cost_usd)}</td>
                      <td className="text-right tabular-nums">{fmtInt(m.turns)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <div className="card" data-testid="usage-top-agents">
            <div className="text-xs uppercase tracking-wide text-ink-faint mb-2">
              {t("usage.top_agents")}
            </div>
            {topAgents.length === 0 ? (
              <div
                className="text-sm text-ink-faint"
                data-testid="usage-top-agents-empty"
              >
                {t("usage.empty")}
              </div>
            ) : (
              <table className="w-full text-sm">
                <thead className="text-[10px] uppercase tracking-wide text-ink-faint">
                  <tr>
                    <th className="text-left font-normal pb-1">{t("usage.col.agent")}</th>
                    <th className="text-right font-normal pb-1">{t("usage.col.cost")}</th>
                    <th className="text-right font-normal pb-1">{t("usage.col.turns")}</th>
                  </tr>
                </thead>
                <tbody>
                  {topAgents.map((a: UsageByAgent) => (
                    <tr key={a.agent_id} data-testid={`usage-top-agent-${a.agent_id}`}>
                      <td className="truncate pr-2">{a.agent_id || t("usage.no_agent")}</td>
                      <td className="text-right tabular-nums">{fmtUsd(a.cost_usd)}</td>
                      <td className="text-right tabular-nums">{fmtInt(a.turns)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
