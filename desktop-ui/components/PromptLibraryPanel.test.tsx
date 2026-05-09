import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { PromptLibraryPanel } from "./PromptLibraryPanel";
import { useAppStore } from "@/stores/appStore";
import type { PromptTemplate } from "@/api/client";

vi.mock("@/api/client", () => ({
  PromptTemplates: {
    list: vi.fn(),
    get: vi.fn(),
    create: vi.fn(),
    update: vi.fn(),
    delete: vi.fn(),
    use: vi.fn(),
  },
  Settings: {
    save: vi.fn(),
  },
}));

import { PromptTemplates, Settings } from "@/api/client";

const READY_STATUS = {
  status: "ready" as const,
  port: 4242,
  token: "test-token",
};

function _row(overrides: Partial<PromptTemplate> = {}): PromptTemplate {
  return {
    id: "id-default",
    title: "Default title",
    body: "Default body",
    kind: "snippet",
    tags: "",
    created_at: "2026-05-09T00:00:00Z",
    updated_at: "2026-05-09T00:00:00Z",
    use_count: 0,
    ...overrides,
  };
}

const SEED: PromptTemplate[] = [
  _row({
    id: "snip-a",
    title: "Greeting",
    body: "Hello there!",
    kind: "snippet",
    tags: "greeting,hello",
    use_count: 4,
  }),
  _row({
    id: "snip-b",
    title: "Email opener",
    body: "Dear team,",
    kind: "snippet",
    tags: "email",
    use_count: 1,
  }),
  _row({
    id: "sys-a",
    title: "Friendly assistant",
    body: "You are a friendly helper.",
    kind: "system_prompt",
    tags: "tone",
    use_count: 0,
  }),
];

const RESET_STATE = useAppStore.getState();

beforeEach(() => {
  useAppStore.setState(
    {
      sidecarStatus: READY_STATUS,
      toasts: [],
      promptTemplates: [],
    },
    false,
  );
  vi.mocked(PromptTemplates.list).mockReset();
  vi.mocked(PromptTemplates.list).mockResolvedValue(SEED);
  vi.mocked(PromptTemplates.create).mockReset();
  vi.mocked(PromptTemplates.update).mockReset();
  vi.mocked(PromptTemplates.delete).mockReset();
  vi.mocked(PromptTemplates.use).mockReset();
  vi.mocked(Settings.save).mockReset();
  vi.mocked(Settings.save).mockResolvedValue({ ok: true });
});

afterEach(() => {
  cleanup();
  useAppStore.setState(RESET_STATE, true);
  vi.restoreAllMocks();
});

async function renderPanel(): Promise<void> {
  render(<PromptLibraryPanel />);
  await waitFor(() => {
    expect(PromptTemplates.list).toHaveBeenCalled();
  });
  await waitFor(() => {
    expect(screen.getByTestId("prompt-row-snip-a")).toBeTruthy();
  });
}

describe("PromptLibraryPanel rendering", () => {
  it("renders all templates from the backend", async () => {
    await renderPanel();
    expect(screen.getByTestId("prompt-row-snip-a")).toBeTruthy();
    expect(screen.getByTestId("prompt-row-snip-b")).toBeTruthy();
    expect(screen.getByTestId("prompt-row-sys-a")).toBeTruthy();
  });
});

describe("PromptLibraryPanel filter by kind", () => {
  it("snippets filter hides system_prompt rows", async () => {
    await renderPanel();
    await userEvent.click(screen.getByTestId("prompt-filter-snippet"));
    expect(screen.getByTestId("prompt-row-snip-a")).toBeTruthy();
    expect(screen.getByTestId("prompt-row-snip-b")).toBeTruthy();
    expect(screen.queryByTestId("prompt-row-sys-a")).toBeNull();
  });

  it("system_prompts filter hides snippet rows", async () => {
    await renderPanel();
    await userEvent.click(screen.getByTestId("prompt-filter-system_prompt"));
    expect(screen.getByTestId("prompt-row-sys-a")).toBeTruthy();
    expect(screen.queryByTestId("prompt-row-snip-a")).toBeNull();
  });
});

describe("PromptLibraryPanel search", () => {
  it("filters by title substring", async () => {
    await renderPanel();
    const search = screen.getByTestId("prompt-search-input") as HTMLInputElement;
    await userEvent.type(search, "Email");
    expect(screen.getByTestId("prompt-row-snip-b")).toBeTruthy();
    expect(screen.queryByTestId("prompt-row-snip-a")).toBeNull();
    expect(screen.queryByTestId("prompt-row-sys-a")).toBeNull();
  });

  it("filters by tag substring", async () => {
    await renderPanel();
    const search = screen.getByTestId("prompt-search-input") as HTMLInputElement;
    await userEvent.type(search, "greeting");
    expect(screen.getByTestId("prompt-row-snip-a")).toBeTruthy();
    expect(screen.queryByTestId("prompt-row-snip-b")).toBeNull();
  });
});

describe("PromptLibraryPanel edit modal", () => {
  it("clicking a row opens the edit modal with prefilled values", async () => {
    await renderPanel();
    await userEvent.click(screen.getByTestId("prompt-row-snip-a"));
    expect(screen.getByTestId("prompt-edit-modal")).toBeTruthy();
    const title = screen.getByTestId("prompt-edit-title") as HTMLInputElement;
    const body = screen.getByTestId("prompt-edit-body") as HTMLTextAreaElement;
    const tags = screen.getByTestId("prompt-edit-tags") as HTMLInputElement;
    expect(title.value).toBe("Greeting");
    expect(body.value).toBe("Hello there!");
    expect(tags.value).toBe("greeting,hello");
  });

  it("save invokes updatePromptTemplate with edited values", async () => {
    vi.mocked(PromptTemplates.update).mockResolvedValue(
      _row({
        id: "snip-a",
        title: "Greeting (v2)",
        body: "Hello there!",
        kind: "snippet",
        tags: "greeting,hello",
      }),
    );
    await renderPanel();
    await userEvent.click(screen.getByTestId("prompt-row-snip-a"));

    const title = screen.getByTestId("prompt-edit-title") as HTMLInputElement;
    await userEvent.clear(title);
    await userEvent.type(title, "Greeting (v2)");

    await userEvent.click(screen.getByTestId("prompt-edit-save"));

    await waitFor(() => {
      expect(PromptTemplates.update).toHaveBeenCalledTimes(1);
    });
    expect(vi.mocked(PromptTemplates.update).mock.calls[0][0]).toBe("snip-a");
    const payload = vi.mocked(PromptTemplates.update).mock.calls[0][1];
    expect(payload.title).toBe("Greeting (v2)");
    expect(payload.body).toBe("Hello there!");
    expect(payload.kind).toBe("snippet");

    // Modal closes after a successful save.
    await waitFor(() => {
      expect(screen.queryByTestId("prompt-edit-modal")).toBeNull();
    });
  });

  it("New prompt button opens the modal in create mode", async () => {
    vi.mocked(PromptTemplates.create).mockResolvedValue(
      _row({
        id: "fresh-id",
        title: "Fresh",
        body: "Fresh body",
        kind: "snippet",
      }),
    );
    await renderPanel();
    await userEvent.click(screen.getByTestId("prompt-new-button"));

    const title = screen.getByTestId("prompt-edit-title") as HTMLInputElement;
    expect(title.value).toBe("");
    await userEvent.type(title, "Fresh");
    await userEvent.type(
      screen.getByTestId("prompt-edit-body"),
      "Fresh body",
    );
    await userEvent.click(screen.getByTestId("prompt-edit-save"));

    await waitFor(() => {
      expect(PromptTemplates.create).toHaveBeenCalledTimes(1);
    });
    const payload = vi.mocked(PromptTemplates.create).mock.calls[0][0];
    expect(payload.title).toBe("Fresh");
    expect(payload.body).toBe("Fresh body");
    expect(payload.kind).toBe("snippet");
  });
});

describe("PromptLibraryPanel delete", () => {
  it("delete with confirm calls deletePromptTemplate", async () => {
    vi.mocked(PromptTemplates.delete).mockResolvedValue({ ok: true });
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    await renderPanel();

    await userEvent.click(screen.getByTestId("prompt-delete-snip-a"));

    expect(confirmSpy).toHaveBeenCalled();
    await waitFor(() => {
      expect(PromptTemplates.delete).toHaveBeenCalledWith("snip-a");
    });
    // Row removed from the rendered list.
    await waitFor(() => {
      expect(screen.queryByTestId("prompt-row-snip-a")).toBeNull();
    });
  });

  it("dismissing the confirm prompt does not call delete", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(false);
    await renderPanel();

    await userEvent.click(screen.getByTestId("prompt-delete-snip-a"));
    expect(PromptTemplates.delete).not.toHaveBeenCalled();
    expect(screen.getByTestId("prompt-row-snip-a")).toBeTruthy();
  });
});

describe("PromptLibraryPanel set default system prompt", () => {
  it("clicking 'Set as default' on a system_prompt writes Settings.save with the body", async () => {
    await renderPanel();
    await userEvent.click(screen.getByTestId("prompt-set-default-sys-a"));
    await waitFor(() => {
      expect(Settings.save).toHaveBeenCalledWith(
        "system_prompt",
        "You are a friendly helper.",
      );
    });
  });
});
