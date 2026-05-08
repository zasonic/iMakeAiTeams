import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { UpdateBanner } from "./UpdateBanner";
import { useAppStore } from "@/stores/appStore";

const RESET_STATE = useAppStore.getState();

beforeEach(() => {
  useAppStore.setState(
    {
      updateReady: null,
      updateBannerDismissed: false,
    },
    false,
  );
  // jsdom doesn't ship a window.electronAPI; stub the bits the banner uses.
  // onUpdateDownloaded returns its unsubscribe handle, mirroring the real
  // preload bridge contract so the effect's cleanup path doesn't blow up.
  Object.assign(window, {
    electronAPI: {
      onUpdateDownloaded: vi.fn().mockReturnValue(() => {}),
      installUpdate: vi.fn().mockResolvedValue(undefined),
    },
  });
});

afterEach(() => {
  cleanup();
  useAppStore.setState(RESET_STATE, true);
});

describe("UpdateBanner", () => {
  it("does not render when updateReady is null", () => {
    render(<UpdateBanner />);
    expect(screen.queryByTestId("update-banner")).toBeNull();
  });

  it("renders banner text including the version when updateReady is set", () => {
    useAppStore.setState({ updateReady: { version: "5.1.0" } });
    render(<UpdateBanner />);
    expect(screen.getByTestId("update-banner")).toBeTruthy();
    expect(screen.getByText(/5\.1\.0/)).toBeTruthy();
    expect(screen.getByText(/Restart to apply/i)).toBeTruthy();
  });

  it("clicking 'Restart now' calls window.electronAPI.installUpdate", async () => {
    useAppStore.setState({ updateReady: { version: "5.1.0" } });
    render(<UpdateBanner />);

    await userEvent.click(screen.getByRole("button", { name: /restart now/i }));
    expect(window.electronAPI.installUpdate).toHaveBeenCalledTimes(1);
  });

  it("clicking 'Later' sets updateBannerDismissed and hides the banner", async () => {
    useAppStore.setState({ updateReady: { version: "5.1.0" } });
    render(<UpdateBanner />);

    await userEvent.click(screen.getByRole("button", { name: /later/i }));
    expect(useAppStore.getState().updateBannerDismissed).toBe(true);
    expect(screen.queryByTestId("update-banner")).toBeNull();
  });
});
