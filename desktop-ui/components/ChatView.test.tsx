import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ChatView } from "./ChatView";
import { useAppStore } from "@/stores/appStore";

// ChatView pulls every backend surface it touches through @/api/client.
// Mock just the methods the export menu exercises plus the bootstrapping
// reads (list / messages / Settings.get) so the component mounts without
// hitting the network. The mock factory has to enumerate every export the
// component imports, otherwise vitest replaces them with `undefined`.
vi.mock("@/api/client", () => ({
  Chat: {
    list: vi.fn(),
    messages: vi.fn(),
    exportConversation: vi.fn(),
    send: vi.fn(),
    stop: vi.fn(),
    newConversation: vi.fn(),
  },
  Docker: {
    classify: vi.fn(),
    health: vi.fn(),
    execute: vi.fn(),
    cancel: vi.fn(),
    approve: vi.fn(),
  },
  Settings: {
    get: vi.fn(),
  },
  Attachments: {
    list: vi.fn(),
    upload: vi.fn(),
    delete: vi.fn(),
  },
}));

import { Attachments, Chat, Settings } from "@/api/client";

// jsdom doesn't ship ResizeObserver but the virtualized message list
// instantiates one. A no-op stand-in is enough — the export menu doesn't
// care about layout measurements.
class _RO {
  observe() {}
  unobserve() {}
  disconnect() {}
}

const READY_STATUS = {
  status: "ready" as const,
  port: 1234,
  token: "test-token",
};

const FAKE_CONVERSATIONS = [
  { id: "conv-A", title: "Alpha chat", updated_at: "2026-05-01T10:00:00Z" },
];

const FAKE_MESSAGES = [
  { id: "m1", role: "user", content: "hi" },
  { id: "m2", role: "assistant", content: "hello" },
];

const RESET_STATE = useAppStore.getState();

beforeEach(() => {
  Object.assign(globalThis, { ResizeObserver: _RO });
  useAppStore.setState(
    {
      sidecarStatus: READY_STATUS,
      toasts: [],
      powerModeRuns: {},
      pendingAttachments: {},
    },
    false,
  );
  Object.assign(window, {
    electronAPI: {
      saveFileDialog: vi
        .fn()
        .mockResolvedValue({ ok: true, path: "/tmp/out.md" }),
      exportPdf: vi
        .fn()
        .mockResolvedValue({ ok: true, path: "/tmp/out.pdf" }),
    },
  });
  vi.mocked(Chat.list).mockResolvedValue(FAKE_CONVERSATIONS);
  vi.mocked(Chat.messages).mockResolvedValue(FAKE_MESSAGES);
  vi.mocked(Chat.exportConversation).mockReset();
  vi.mocked(Settings.get).mockResolvedValue({ power_mode_enabled: false } as never);
  vi.mocked(Attachments.list).mockResolvedValue([]);
  vi.mocked(Attachments.upload).mockReset();
  vi.mocked(Attachments.delete).mockReset();
});

afterEach(() => {
  cleanup();
  useAppStore.setState(RESET_STATE, true);
});

async function renderWithLoadedConversation(): Promise<void> {
  render(<ChatView />);
  await waitFor(() => {
    expect(screen.getByTestId("chat-export-button")).toBeTruthy();
  });
  // The button is disabled until messages land. Wait for the load.
  await waitFor(() => {
    const btn = screen.getByTestId("chat-export-button") as HTMLButtonElement;
    expect(btn.disabled).toBe(false);
  });
}

describe("ChatView export menu", () => {
  it("opens the export menu on click", async () => {
    await renderWithLoadedConversation();

    expect(screen.queryByTestId("chat-export-menu")).toBeNull();
    await userEvent.click(screen.getByTestId("chat-export-button"));
    expect(screen.getByTestId("chat-export-menu")).toBeTruthy();
    expect(screen.getByTestId("chat-export-md")).toBeTruthy();
    expect(screen.getByTestId("chat-export-json")).toBeTruthy();
    expect(screen.getByTestId("chat-export-pdf")).toBeTruthy();
  });

  it("Markdown click invokes exportConversation with the md format and saves via saveFileDialog", async () => {
    vi.mocked(Chat.exportConversation).mockResolvedValue("# md body");
    await renderWithLoadedConversation();

    await userEvent.click(screen.getByTestId("chat-export-button"));
    await userEvent.click(screen.getByTestId("chat-export-md"));

    await waitFor(() => {
      expect(Chat.exportConversation).toHaveBeenCalledWith("conv-A", "md");
    });
    await waitFor(() => {
      expect(window.electronAPI.saveFileDialog).toHaveBeenCalledTimes(1);
    });
    const [name, body] = vi.mocked(window.electronAPI.saveFileDialog).mock.calls[0];
    expect(name).toBe("Alpha chat.md");
    expect(body).toBe("# md body");
  });

  it("JSON click invokes exportConversation with the json format", async () => {
    vi.mocked(Chat.exportConversation).mockResolvedValue("[]");
    await renderWithLoadedConversation();

    await userEvent.click(screen.getByTestId("chat-export-button"));
    await userEvent.click(screen.getByTestId("chat-export-json"));

    await waitFor(() => {
      expect(Chat.exportConversation).toHaveBeenCalledWith("conv-A", "json");
    });
    await waitFor(() => {
      const calls = vi.mocked(window.electronAPI.saveFileDialog).mock.calls;
      expect(calls.length).toBe(1);
      expect(calls[0][0]).toBe("Alpha chat.json");
      expect(calls[0][1]).toBe("[]");
    });
  });

  it("PDF click fetches pdf-html and forwards the HTML to electronAPI.exportPdf", async () => {
    vi.mocked(Chat.exportConversation).mockResolvedValue(
      "<html><body>chat</body></html>",
    );
    await renderWithLoadedConversation();

    await userEvent.click(screen.getByTestId("chat-export-button"));
    await userEvent.click(screen.getByTestId("chat-export-pdf"));

    await waitFor(() => {
      expect(Chat.exportConversation).toHaveBeenCalledWith(
        "conv-A",
        "pdf-html",
      );
    });
    await waitFor(() => {
      expect(window.electronAPI.exportPdf).toHaveBeenCalledTimes(1);
    });
    const [html, name] = vi.mocked(window.electronAPI.exportPdf).mock.calls[0];
    expect(html).toBe("<html><body>chat</body></html>");
    expect(name).toBe("Alpha chat.pdf");
  });
});

// ── Attachments (PR 8) ─────────────────────────────────────────────────────

function _makeFile(name: string, content = "hello"): File {
  // jsdom's File constructor accepts (parts, name, options).
  return new File([content], name, { type: "text/plain" });
}

function _dataTransfer(files: File[], shiftKey: boolean) {
  // Build a FileList-shaped object: index access + length + item().
  const fileList: Record<string, unknown> = {
    length: files.length,
    item: (i: number) => files[i] ?? null,
  };
  files.forEach((f, i) => {
    fileList[i] = f;
  });
  return {
    types: ["Files"],
    files: fileList,
    dropEffect: "",
    effectAllowed: "all",
    getData: () => "",
    setData: () => {},
    shiftKey,
  };
}

function _fireDragEvent(target: Element, type: string, files: File[], shiftKey: boolean) {
  const dt = _dataTransfer(files, shiftKey);
  const event = new Event(type, { bubbles: true, cancelable: true });
  Object.defineProperty(event, "dataTransfer", { value: dt });
  Object.defineProperty(event, "shiftKey", { value: shiftKey });
  target.dispatchEvent(event);
}

describe("ChatView attachments", () => {
  it("dragover shows the drop overlay", async () => {
    await renderWithLoadedConversation();
    expect(screen.queryByTestId("chat-drop-overlay")).toBeNull();

    const target = screen.getByTestId("chat-drop-target");
    _fireDragEvent(target, "dragenter", [_makeFile("a.txt")], false);

    await waitFor(() => {
      expect(screen.getByTestId("chat-drop-overlay")).toBeTruthy();
    });
  });

  it("drop without Shift uploads with persist=false", async () => {
    vi.mocked(Attachments.upload).mockResolvedValue({
      id: "att-1",
      filename: "a.txt",
      size_bytes: 5,
      persist: false,
      extract_chars: 5,
    });
    await renderWithLoadedConversation();

    const target = screen.getByTestId("chat-drop-target");
    _fireDragEvent(target, "dragenter", [_makeFile("a.txt")], false);
    _fireDragEvent(target, "drop", [_makeFile("a.txt")], false);

    await waitFor(() => {
      expect(Attachments.upload).toHaveBeenCalledTimes(1);
    });
    const call = vi.mocked(Attachments.upload).mock.calls[0];
    expect(call[0]).toBe("conv-A");
    expect((call[1] as File).name).toBe("a.txt");
    expect(call[2]).toBe(false);
  });

  it("drop with Shift uploads with persist=true", async () => {
    vi.mocked(Attachments.upload).mockResolvedValue({
      id: "att-2",
      filename: "doc.md",
      size_bytes: 9,
      persist: true,
      extract_chars: 9,
    });
    await renderWithLoadedConversation();

    const target = screen.getByTestId("chat-drop-target");
    _fireDragEvent(target, "dragenter", [_makeFile("doc.md")], true);
    _fireDragEvent(target, "drop", [_makeFile("doc.md")], true);

    await waitFor(() => {
      expect(Attachments.upload).toHaveBeenCalledTimes(1);
    });
    expect(vi.mocked(Attachments.upload).mock.calls[0][2]).toBe(true);
  });

  it("clicking a chip's X calls Attachments.delete", async () => {
    const seeded = {
      id: "att-3",
      conversation_id: "conv-A",
      filename: "old.txt",
      mime_type: "text/plain",
      size_bytes: 12,
      persist: false,
      rag_doc_id: null,
      created_at: "2026-05-01T10:00:00Z",
    };
    vi.mocked(Attachments.list).mockResolvedValue([seeded]);
    vi.mocked(Attachments.delete).mockResolvedValue({ ok: true });
    await renderWithLoadedConversation();

    await waitFor(() => {
      expect(screen.getByTestId("chat-attachment-chip-att-3")).toBeTruthy();
    });

    await userEvent.click(screen.getByTestId("chat-attachment-remove-att-3"));

    await waitFor(() => {
      expect(Attachments.delete).toHaveBeenCalledWith("att-3");
    });
  });

  it("renders no chip strip when pendingAttachments is empty", async () => {
    await renderWithLoadedConversation();
    expect(screen.queryByTestId("chat-attachment-chips")).toBeNull();
  });
});
