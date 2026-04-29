import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useAppStore } from "./appStore";

const RESET_STATE = useAppStore.getState();

beforeEach(() => {
  vi.useFakeTimers();
  // Snapshot reset so each test starts from a clean store.
  useAppStore.setState({
    toasts: [],
    activeChat: null,
    powerModeRuns: {},
  }, false);
});

afterEach(() => {
  vi.useRealTimers();
  // Restore the original action references in case anything tampered with them.
  useAppStore.setState(RESET_STATE, true);
});

describe("pushToast", () => {
  it("adds a toast and auto-dismisses it after the configured delay", () => {
    useAppStore.getState().pushToast({ kind: "info", text: "hello" });
    expect(useAppStore.getState().toasts).toHaveLength(1);
    expect(useAppStore.getState().toasts[0].text).toBe("hello");

    // Info toasts auto-dismiss after 4000 ms (errors/warnings stick for 8000).
    vi.advanceTimersByTime(3999);
    expect(useAppStore.getState().toasts).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(useAppStore.getState().toasts).toHaveLength(0);
  });

  it("keeps error toasts visible for the longer 8000 ms window", () => {
    useAppStore.getState().pushToast({ kind: "error", text: "boom" });
    vi.advanceTimersByTime(4000);
    expect(useAppStore.getState().toasts).toHaveLength(1);
    vi.advanceTimersByTime(4000);
    expect(useAppStore.getState().toasts).toHaveLength(0);
  });
});

describe("appendChatToken", () => {
  it("trims the streaming buffer once it exceeds the 1 000 000 char cap", () => {
    useAppStore.getState().startChatStream("conv-1");

    // Just under the MAX. Append should not trim.
    useAppStore.getState().appendChatToken("a".repeat(999_999));
    expect(useAppStore.getState().activeChat?.buffer.length).toBe(999_999);

    // Crossing MAX (next.length === 1_000_001) should drop the head and keep
    // KEEP=500_000 chars from the tail.
    useAppStore.getState().appendChatToken("bb");
    expect(useAppStore.getState().activeChat?.buffer.length).toBe(500_000);
    // The kept tail should end in the most recently appended bytes.
    expect(useAppStore.getState().activeChat?.buffer.endsWith("bb")).toBe(true);
  });
});

describe("upsertPowerModeStep", () => {
  it("updates an existing step in place without duplicating it", () => {
    useAppStore.getState().startPowerModeRun("task-1", "conv-1");
    useAppStore.getState().upsertPowerModeStep("task-1", {
      step_id: "s1",
      kind: "shell",
      status: "running",
      command: "echo hi",
    });
    let run = useAppStore.getState().powerModeRuns["task-1"];
    expect(run.steps).toHaveLength(1);
    expect(run.steps[0].status).toBe("running");

    useAppStore.getState().upsertPowerModeStep("task-1", {
      step_id: "s1",
      kind: "shell",
      status: "done",
      stdout: "hi",
    });
    run = useAppStore.getState().powerModeRuns["task-1"];
    expect(run.steps).toHaveLength(1);
    expect(run.steps[0].status).toBe("done");
    expect(run.steps[0].stdout).toBe("hi");
    // Earlier fields from the running step are preserved through the merge.
    expect(run.steps[0].command).toBe("echo hi");
  });
});

describe("endPowerModeRun cleanup", () => {
  it("filters expired approvals and removes the run after 2 hours", () => {
    const now = 1_700_000_000_000;
    vi.setSystemTime(now);
    useAppStore.getState().startPowerModeRun("task-2", "conv-1");
    useAppStore.getState().addPowerModeApproval("task-2", {
      approval_id: "a-fresh",
      summary: "fresh",
      details: {},
      danger: "low",
      expires_at: now + 60_000,
    });
    useAppStore.getState().addPowerModeApproval("task-2", {
      approval_id: "a-expired",
      summary: "expired",
      details: {},
      danger: "low",
      expires_at: now - 1,
    });

    useAppStore.getState().endPowerModeRun("task-2");
    const ended = useAppStore.getState().powerModeRuns["task-2"];
    expect(ended.done).toBe(true);
    expect(ended.approvals.map((a) => a.approval_id)).toEqual(["a-fresh"]);

    // Just past the 2-hour TTL the completed run is swept out of state.
    // vi's fake timer queue advances independently of the mocked Date, so
    // bump the system clock forward and then drain the timer queue.
    vi.setSystemTime(now + 7_200_001);
    vi.advanceTimersByTime(7_200_001);
    expect(useAppStore.getState().powerModeRuns["task-2"]).toBeUndefined();
  });
});
