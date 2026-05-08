import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { UsagePanel } from "./UsagePanel";
import { useAppStore } from "@/stores/appStore";

vi.mock("@/api/client", () => ({
  Usage: {
    getUsageSummary: vi.fn(),
  },
}));

// recharts touches DOM measurement APIs (ResponsiveContainer queries
// offsetWidth) that jsdom doesn't implement, so we replace it with a
// flat fan-out that produces one node per data point. This lets us assert
// "a bar per row" without running real layout.
vi.mock("recharts", () => {
  const Pass = ({ children }: { children?: React.ReactNode }) => (
    <div>{children}</div>
  );
  return {
    ResponsiveContainer: Pass,
    BarChart: ({ children, data }: { children?: React.ReactNode; data: unknown[] }) => (
      <div data-testid="rc-bar-chart">
        {Array.isArray(data) &&
          data.map((d, i) => {
            const key = (d as { key?: string }).key ?? `row-${i}`;
            return <div key={i} data-testid={`rc-bar-${key}`} />;
          })}
        {children}
      </div>
    ),
    Bar:            () => null,
    XAxis:          () => null,
    YAxis:          () => null,
    CartesianGrid:  () => null,
    Tooltip:        () => null,
  };
});

import { Usage } from "@/api/client";
import type { UsageSummary } from "@/types/usage";

const READY_STATUS = {
  status: "ready" as const,
  port:   1234,
  token:  "test-token",
};

const SEEDED_SUMMARY: UsageSummary = {
  window_days: 30,
  group_by:    "day",
  total: {
    input_tokens:  12345,
    output_tokens: 6789,
    cost_usd:      1.2345,
    turns:         42,
  },
  rows: [
    { key: "2026-04-30", input_tokens: 100, output_tokens: 50, cost_usd: 0.10, turns: 1 },
    { key: "2026-05-01", input_tokens: 200, output_tokens: 80, cost_usd: 0.20, turns: 2 },
    { key: "2026-05-02", input_tokens: 300, output_tokens: 120, cost_usd: 0.30, turns: 3 },
  ],
  by_model: [
    { model: "claude-opus", cost_usd: 0.80, turns: 10 },
    { model: "qwen3",       cost_usd: 0.20, turns: 30 },
  ],
  by_agent: [
    { agent_id: "agent-A", cost_usd: 0.60, turns: 8 },
    { agent_id: "agent-B", cost_usd: 0.40, turns: 12 },
  ],
};

const EMPTY_SUMMARY: UsageSummary = {
  window_days: 30,
  group_by:    "day",
  total: { input_tokens: 0, output_tokens: 0, cost_usd: 0, turns: 0 },
  rows: [],
  by_model: [],
  by_agent: [],
};

const RESET_STATE = useAppStore.getState();

beforeEach(() => {
  useAppStore.setState({ sidecarStatus: READY_STATUS, toasts: [] }, false);
  vi.mocked(Usage.getUsageSummary).mockReset();
});

afterEach(() => {
  cleanup();
  useAppStore.setState(RESET_STATE, true);
});

describe("UsagePanel", () => {
  it("renders all stat cards with mocked summary data", async () => {
    vi.mocked(Usage.getUsageSummary).mockResolvedValue(SEEDED_SUMMARY);

    render(<UsagePanel />);

    await waitFor(() => {
      expect(screen.getByTestId("usage-stat-cost")).toBeTruthy();
    });
    expect(screen.getByTestId("usage-stat-turns")).toBeTruthy();
    expect(screen.getByTestId("usage-stat-input")).toBeTruthy();
    expect(screen.getByTestId("usage-stat-output")).toBeTruthy();

    // Spot-check the headline numbers rendered with locale formatting.
    expect(screen.getByTestId("usage-stat-turns").textContent).toContain("42");
    expect(screen.getByTestId("usage-stat-input").textContent).toContain("12,345");
    expect(screen.getByTestId("usage-stat-output").textContent).toContain("6,789");
    // cost is rendered through toLocaleString currency, which always has
    // a dollar sign for en-US locale; we only assert the digits.
    expect(screen.getByTestId("usage-stat-cost").textContent).toMatch(/1\.23/);

    // Top tables surface the seeded entries.
    expect(screen.getByTestId("usage-top-model-claude-opus")).toBeTruthy();
    expect(screen.getByTestId("usage-top-model-qwen3")).toBeTruthy();
    expect(screen.getByTestId("usage-top-agent-agent-A")).toBeTruthy();
    expect(screen.getByTestId("usage-top-agent-agent-B")).toBeTruthy();
  });

  it("changing the time window triggers a refetch with the new days param", async () => {
    vi.mocked(Usage.getUsageSummary).mockResolvedValue(SEEDED_SUMMARY);

    render(<UsagePanel />);
    await waitFor(() => {
      expect(Usage.getUsageSummary).toHaveBeenCalledWith(30, "day");
    });
    vi.mocked(Usage.getUsageSummary).mockClear();

    await userEvent.click(screen.getByTestId("usage-window-7"));

    await waitFor(() => {
      expect(Usage.getUsageSummary).toHaveBeenCalledWith(7, "day");
    });
  });

  it("changing the group-by triggers a refetch with the new group_by param", async () => {
    vi.mocked(Usage.getUsageSummary).mockResolvedValue(SEEDED_SUMMARY);

    render(<UsagePanel />);
    await waitFor(() => {
      expect(Usage.getUsageSummary).toHaveBeenCalledWith(30, "day");
    });
    vi.mocked(Usage.getUsageSummary).mockClear();

    await userEvent.click(screen.getByTestId("usage-group-model"));

    await waitFor(() => {
      expect(Usage.getUsageSummary).toHaveBeenCalledWith(30, "model");
    });
  });

  it("renders empty-state strings (not blank chart areas) for an empty summary", async () => {
    vi.mocked(Usage.getUsageSummary).mockResolvedValue(EMPTY_SUMMARY);

    render(<UsagePanel />);

    await waitFor(() => {
      expect(screen.getByTestId("usage-chart-empty")).toBeTruthy();
    });
    expect(screen.getByTestId("usage-top-models-empty")).toBeTruthy();
    expect(screen.getByTestId("usage-top-agents-empty")).toBeTruthy();

    expect(screen.getByTestId("usage-chart-empty").textContent)
      .toContain("No usage in the selected window");
  });

  it("bar chart renders a bar per row", async () => {
    vi.mocked(Usage.getUsageSummary).mockResolvedValue(SEEDED_SUMMARY);

    render(<UsagePanel />);

    await waitFor(() => {
      expect(screen.getByTestId("rc-bar-chart")).toBeTruthy();
    });

    // The mocked recharts BarChart fans out one bar node per data row,
    // keyed by the row's `key` string. SEEDED_SUMMARY has 3 rows.
    expect(screen.getByTestId("rc-bar-2026-04-30")).toBeTruthy();
    expect(screen.getByTestId("rc-bar-2026-05-01")).toBeTruthy();
    expect(screen.getByTestId("rc-bar-2026-05-02")).toBeTruthy();
  });
});
