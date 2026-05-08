import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ModelBrowser } from "./ModelBrowser";
import { useAppStore } from "@/stores/appStore";

vi.mock("@/api/client", () => {
  return {
    System: {
      listLocalModels: vi.fn(),
      setActiveLocalModel: vi.fn(),
    },
  };
});

import { System, type LocalModelRow } from "@/api/client";

const READY_STATUS = {
  status: "ready" as const,
  port: 1234,
  token: "test-token",
};

const SAMPLE_MODELS: LocalModelRow[] = [
  {
    id:             "qwen3-30b-a3b-q4",
    size_bytes:     8_500_000_000,
    context_length: 32_768,
    quantization:   "Q4_K_M",
    backend:        "ollama",
    loaded:         false,
  },
  {
    id:             "lmstudio-community/Qwen3-7B",
    size_bytes:     null,
    context_length: null,
    quantization:   null,
    backend:        "lm_studio",
    loaded:         false,
  },
];

const RESET_STATE = useAppStore.getState();

beforeEach(() => {
  useAppStore.setState(
    { sidecarStatus: READY_STATUS, toasts: [] },
    false,
  );
  vi.mocked(System.listLocalModels).mockReset();
  vi.mocked(System.setActiveLocalModel).mockReset();
});

afterEach(() => {
  cleanup();
  useAppStore.setState(RESET_STATE, true);
});

describe("ModelBrowser", () => {
  it("renders empty state when models list is empty", async () => {
    vi.mocked(System.listLocalModels).mockResolvedValue({
      models:  [],
      current: "",
    });

    render(<ModelBrowser />);

    await waitFor(() => {
      expect(screen.getByTestId("model-empty")).toBeTruthy();
    });
    expect(screen.queryByTestId("model-row")).toBeNull();
  });

  it("renders rows when models list has items", async () => {
    vi.mocked(System.listLocalModels).mockResolvedValue({
      models:  SAMPLE_MODELS,
      current: "",
    });

    render(<ModelBrowser />);

    await waitFor(() => {
      expect(screen.getAllByTestId("model-row")).toHaveLength(2);
    });
    expect(screen.getByText("qwen3-30b-a3b-q4")).toBeTruthy();
    expect(screen.getByText("lmstudio-community/Qwen3-7B")).toBeTruthy();
    expect(screen.queryByTestId("model-empty")).toBeNull();
  });

  it("clicking 'Use this model' calls setActiveLocalModel with the right id", async () => {
    vi.mocked(System.listLocalModels).mockResolvedValue({
      models:  SAMPLE_MODELS,
      current: "",
    });
    vi.mocked(System.setActiveLocalModel).mockResolvedValue({
      current: "qwen3-30b-a3b-q4",
      ok:      true,
    });

    render(<ModelBrowser />);
    await waitFor(() => {
      expect(screen.getAllByTestId("model-row")).toHaveLength(2);
    });

    const rows = screen.getAllByTestId("model-row");
    const firstRowButton = within(rows[0]).getByRole("button", {
      name: /use this model/i,
    });
    await userEvent.click(firstRowButton);

    expect(System.setActiveLocalModel).toHaveBeenCalledWith("qwen3-30b-a3b-q4");
  });

  it("shows 'Active' badge on the model whose id matches `current`", async () => {
    vi.mocked(System.listLocalModels).mockResolvedValue({
      models:  SAMPLE_MODELS,
      current: "lmstudio-community/Qwen3-7B",
    });

    render(<ModelBrowser />);

    await waitFor(() => {
      expect(screen.getAllByTestId("model-row")).toHaveLength(2);
    });

    const rows = screen.getAllByTestId("model-row");
    expect(within(rows[0]).queryByTestId("active-badge")).toBeNull();
    expect(within(rows[1]).getByTestId("active-badge")).toBeTruthy();
  });
});
