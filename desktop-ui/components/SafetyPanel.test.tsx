import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { SafetyPanel } from "./SafetyPanel";
import { useAppStore } from "@/stores/appStore";

vi.mock("@/api/client", () => ({
  Safety: {
    getSafetySummary: vi.fn(),
  },
}));

import { Safety } from "@/api/client";
import type { SafetySummary } from "@/types/safety";

const READY_STATUS = {
  status: "ready" as const,
  port:   1234,
  token:  "test-token",
};

const SEEDED_SUMMARY: SafetySummary = {
  window_days: 30,
  escalations: { triggered: 4, approved: 2, denied: 1, pending: 1 },
  memory_gate: {
    facts_proposed: 7,
    auto_accepted:  4,
    user_approved:  1,
    user_denied:    1,
    pending:        1,
  },
  canary: {
    baselines:     3,
    alerts_fired:  2,
    last_alert_at: "2026-05-01T10:30:45.000Z",
  },
  governance: {
    tool_calls_total:   8,
    tool_calls_denied:  5,
    denial_top_reasons: [
      { reason: "tool not in allowlist", count: 3 },
      { reason: "rate limit exceeded",   count: 2 },
    ],
  },
  routing: {
    turns_total:    20,
    turns_failed:   3,
    mast_breakdown: [
      { category: "tool_misuse",    count: 2 },
      { category: "format_violation", count: 1 },
    ],
  },
  voting: { high_stakes_turns: 5, consensus_reached: 4 },
};

const EMPTY_SUMMARY: SafetySummary = {
  window_days: 30,
  escalations: { triggered: 0, approved: 0, denied: 0, pending: 0 },
  memory_gate: {
    facts_proposed: 0,
    auto_accepted:  0,
    user_approved:  0,
    user_denied:    0,
    pending:        0,
  },
  canary:     { baselines: 0, alerts_fired: 0, last_alert_at: null },
  governance: { tool_calls_total: 0, tool_calls_denied: 0, denial_top_reasons: [] },
  routing:    { turns_total: 0, turns_failed: 0, mast_breakdown: [] },
  voting:     { high_stakes_turns: 0, consensus_reached: 0 },
};

const RESET_STATE = useAppStore.getState();

beforeEach(() => {
  useAppStore.setState({ sidecarStatus: READY_STATUS, toasts: [] }, false);
  vi.mocked(Safety.getSafetySummary).mockReset();
});

afterEach(() => {
  cleanup();
  useAppStore.setState(RESET_STATE, true);
});

describe("SafetyPanel", () => {
  it("renders all six cards with mocked summary data", async () => {
    vi.mocked(Safety.getSafetySummary).mockResolvedValue(SEEDED_SUMMARY);

    render(<SafetyPanel />);

    await waitFor(() => {
      expect(screen.getByTestId("safety-card-escalations")).toBeTruthy();
    });

    // All six cards must be present
    expect(screen.getByTestId("safety-card-escalations")).toBeTruthy();
    expect(screen.getByTestId("safety-card-memory")).toBeTruthy();
    expect(screen.getByTestId("safety-card-canary")).toBeTruthy();
    expect(screen.getByTestId("safety-card-governance")).toBeTruthy();
    expect(screen.getByTestId("safety-card-routing")).toBeTruthy();
    expect(screen.getByTestId("safety-card-voting")).toBeTruthy();

    // Spot-check the headline metrics rendered into the cards
    const escCard = screen.getByTestId("safety-card-escalations");
    expect(escCard.textContent).toContain("4");
    const memCard = screen.getByTestId("safety-card-memory");
    expect(memCard.textContent).toContain("7");
    const govCard = screen.getByTestId("safety-card-governance");
    expect(govCard.textContent).toContain("5");
    expect(govCard.textContent).toContain("Tool not in allowlist");
    const routCard = screen.getByTestId("safety-card-routing");
    expect(routCard.textContent).toContain("Tool misuse");
    const voteCard = screen.getByTestId("safety-card-voting");
    expect(voteCard.textContent).toContain("4 / 5");
  });

  it("changing the time window triggers a refetch with the new days param", async () => {
    vi.mocked(Safety.getSafetySummary).mockResolvedValue(SEEDED_SUMMARY);

    render(<SafetyPanel />);
    await waitFor(() => {
      expect(Safety.getSafetySummary).toHaveBeenCalledWith(30);
    });
    vi.mocked(Safety.getSafetySummary).mockClear();

    await userEvent.click(screen.getByTestId("safety-window-7"));

    await waitFor(() => {
      expect(Safety.getSafetySummary).toHaveBeenCalledWith(7);
    });
  });

  it("refresh button refetches the summary", async () => {
    vi.mocked(Safety.getSafetySummary).mockResolvedValue(SEEDED_SUMMARY);

    render(<SafetyPanel />);
    await waitFor(() => {
      expect(Safety.getSafetySummary).toHaveBeenCalledTimes(1);
    });

    await userEvent.click(screen.getByTestId("safety-refresh"));

    await waitFor(() => {
      expect(Safety.getSafetySummary).toHaveBeenCalledTimes(2);
    });
  });

  it("renders empty-state strings (not blank cards) for an empty summary", async () => {
    vi.mocked(Safety.getSafetySummary).mockResolvedValue(EMPTY_SUMMARY);

    render(<SafetyPanel />);

    await waitFor(() => {
      expect(screen.getByTestId("safety-card-escalations-empty")).toBeTruthy();
    });
    expect(screen.getByTestId("safety-card-memory-empty")).toBeTruthy();
    expect(screen.getByTestId("safety-card-canary-empty")).toBeTruthy();
    expect(screen.getByTestId("safety-card-governance-empty")).toBeTruthy();
    expect(screen.getByTestId("safety-card-routing-empty")).toBeTruthy();
    expect(screen.getByTestId("safety-card-voting-empty")).toBeTruthy();

    expect(screen.getByTestId("safety-card-escalations-empty").textContent)
      .toContain("No escalations in the selected window");
  });
});
