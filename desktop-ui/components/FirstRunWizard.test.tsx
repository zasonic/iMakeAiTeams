import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { FirstRunWizard } from "./FirstRunWizard";
import { useAppStore } from "@/stores/appStore";

vi.mock("@/api/client", () => ({
  Settings: {
    verifyApiKey: vi.fn(),
    completeFirstRun: vi.fn(),
    set: vi.fn(),
  },
  System: {
    bundledDownload: vi.fn(),
    bundledStart: vi.fn(),
  },
}));

import { Settings, System } from "@/api/client";

const RESET_STATE = useAppStore.getState();

beforeEach(() => {
  useAppStore.setState(
    {
      toasts: [],
      bundledDownload: {
        status: "idle",
        modelId: "",
        bytesDone: 0,
        bytesTotal: 0,
        error: "",
      },
    },
    false,
  );
  // Stub electronAPI.openExternal for the "Get a key" button. The wizard
  // dispatches it via window.electronAPI which doesn't exist in jsdom.
  Object.assign(window, {
    electronAPI: {
      openExternal: vi.fn().mockResolvedValue(undefined),
    },
  });
  vi.mocked(Settings.verifyApiKey).mockReset();
  vi.mocked(Settings.completeFirstRun).mockReset();
  vi.mocked(Settings.set).mockReset();
  vi.mocked(System.bundledDownload).mockReset();
  vi.mocked(System.bundledStart).mockReset();
});

afterEach(() => {
  cleanup();
  useAppStore.setState(RESET_STATE, true);
});

async function advanceToLocalChoice() {
  vi.mocked(Settings.verifyApiKey).mockResolvedValue({
    ok: true,
    message: "verified",
  });
  render(<FirstRunWizard onComplete={() => {}} />);
  // welcome → claude
  await userEvent.click(screen.getByRole("button", { name: /get started/i }));
  await screen.findByTestId("step-claude");
  await userEvent.type(
    screen.getByPlaceholderText("sk-ant-…"),
    "sk-ant-fake",
  );
  await userEvent.click(
    screen.getByRole("button", { name: /verify & continue/i }),
  );
  await screen.findByTestId("step-local-choice");
}

describe("FirstRunWizard", () => {
  it("renders welcome step on first mount", () => {
    render(<FirstRunWizard onComplete={() => {}} />);
    expect(screen.getByTestId("step-welcome")).toBeTruthy();
  });

  it("walks welcome → claude → local choice", async () => {
    await advanceToLocalChoice();
    expect(screen.getByTestId("choice-quick-start")).toBeTruthy();
    expect(screen.getByTestId("choice-byo")).toBeTruthy();
    expect(screen.getByTestId("choice-skip")).toBeTruthy();
  });

  it("Quick start triggers bundledDownload and renders progress bar", async () => {
    await advanceToLocalChoice();
    vi.mocked(System.bundledDownload).mockResolvedValue({
      ok: true,
      model_id: "Qwen3-4B-Instruct-Q4_K_M",
    });
    await userEvent.click(screen.getByTestId("choice-quick-start"));

    expect(System.bundledDownload).toHaveBeenCalledTimes(1);
    await screen.findByTestId("quick-start-progress");
    // Simulate the SSE-driven store update for a partial download — the
    // progress bar reflects the new bytesDone/bytesTotal in the store.
    useAppStore.setState({
      bundledDownload: {
        status: "downloading",
        modelId: "Qwen3-4B-Instruct-Q4_K_M",
        bytesDone: 500,
        bytesTotal: 1000,
        error: "",
      },
    });
    await waitFor(() => {
      const bar = screen.getByTestId("quick-start-progress");
      expect(bar.getAttribute("aria-valuenow")).toBe("50");
    });
  });

  it("Quick start completion auto-calls bundledStart", async () => {
    await advanceToLocalChoice();
    vi.mocked(System.bundledDownload).mockResolvedValue({ ok: true, model_id: "x" });
    vi.mocked(System.bundledStart).mockResolvedValue({ ok: true, port: 1234 });
    vi.mocked(Settings.set).mockResolvedValue({ ok: true });

    await userEvent.click(screen.getByTestId("choice-quick-start"));

    // Flip the store to "complete" the way the SSE handler would.
    useAppStore.setState({
      bundledDownload: {
        status: "complete",
        modelId: "x",
        bytesDone: 1000,
        bytesTotal: 1000,
        error: "",
      },
    });

    await waitFor(() => {
      expect(System.bundledStart).toHaveBeenCalledWith("x");
    });
    await screen.findByTestId("quick-start-continue");
  });

  it("Skip path completes without hitting any backend", async () => {
    await advanceToLocalChoice();
    vi.mocked(Settings.set).mockResolvedValue({ ok: true });
    vi.mocked(Settings.completeFirstRun).mockResolvedValue({ ok: true });

    await userEvent.click(screen.getByTestId("choice-skip"));
    await screen.findByTestId("step-done");

    expect(System.bundledDownload).not.toHaveBeenCalled();
    expect(System.bundledStart).not.toHaveBeenCalled();

    let onCompleteCalled = false;
    const result = await new Promise<boolean>((resolve) => {
      cleanup();
      render(
        <FirstRunWizard
          onComplete={() => {
            onCompleteCalled = true;
            resolve(true);
          }}
        />,
      );
      // Re-walk straight to done via skip in this fresh render.
      (async () => {
        await userEvent.click(
          screen.getByRole("button", { name: /get started/i }),
        );
        vi.mocked(Settings.verifyApiKey).mockResolvedValue({
          ok: true,
          message: "verified",
        });
        await userEvent.type(
          screen.getByPlaceholderText("sk-ant-…"),
          "sk-ant-fake",
        );
        await userEvent.click(
          screen.getByRole("button", { name: /verify & continue/i }),
        );
        await screen.findByTestId("step-local-choice");
        await userEvent.click(screen.getByTestId("choice-skip"));
        const finishButton = await screen.findByTestId("finish-button");
        await userEvent.click(finishButton);
      })();
    });
    expect(result).toBe(true);
    expect(onCompleteCalled).toBe(true);
  });

  it("BYO path lands on the install instructions", async () => {
    await advanceToLocalChoice();
    await userEvent.click(screen.getByTestId("choice-byo"));
    expect(await screen.findByTestId("step-byo")).toBeTruthy();
  });
});
