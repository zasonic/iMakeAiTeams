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
    searchConversations: vi.fn(),
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
    fetchBlob: vi.fn(),
  },
  Voice: {
    transcribe: vi.fn(),
    synthesize: vi.fn(),
    assetsStatus: vi.fn(),
    assetsDownload: vi.fn(),
  },
  PromptTemplates: {
    list: vi.fn(),
    use: vi.fn(),
  },
}));

import { Attachments, Chat, PromptTemplates, Settings, Voice } from "@/api/client";

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
  vi.mocked(Chat.searchConversations).mockReset();
  vi.mocked(Chat.searchConversations).mockResolvedValue([]);
  vi.mocked(Settings.get).mockResolvedValue({
    power_mode_enabled: false,
    voice_input_enabled: false,
    voice_output_enabled: false,
  } as never);
  vi.mocked(Voice.transcribe).mockReset();
  vi.mocked(Voice.synthesize).mockReset();
  vi.mocked(Attachments.list).mockResolvedValue([]);
  vi.mocked(Attachments.upload).mockReset();
  vi.mocked(Attachments.delete).mockReset();
  // Default: image chips get a tiny opaque blob so the thumbnail effect
  // doesn't crash on a `.then` of undefined. Tests that assert on the
  // thumbnail override this with a richer blob.
  vi.mocked(Attachments.fetchBlob).mockReset();
  vi.mocked(Attachments.fetchBlob).mockResolvedValue(
    new Blob([new Uint8Array([0])], { type: "image/png" }),
  );
  vi.mocked(PromptTemplates.list).mockReset();
  vi.mocked(PromptTemplates.list).mockResolvedValue([]);
  vi.mocked(PromptTemplates.use).mockReset();
  // jsdom doesn't ship URL.createObjectURL — the image chip renders a
  // <img src=…> from one. Stub it (and the matching revoke) so the chip
  // mounts cleanly under test.
  if (typeof URL.createObjectURL !== "function") {
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:fake-thumb"),
    });
  }
  if (typeof URL.revokeObjectURL !== "function") {
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: vi.fn(),
    });
  }
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

// ── PR 11: image input ──────────────────────────────────────────────────────

function _makeImageFile(name = "shot.png", type = "image/png"): File {
  return new File([new Uint8Array([137, 80, 78, 71])], name, { type });
}

function _imageDataTransfer(files: File[], shiftKey: boolean) {
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
    items: files.map((f) => ({
      kind: "file",
      type: f.type,
      getAsFile: () => f,
    })),
    dropEffect: "",
    effectAllowed: "all",
    getData: () => "",
    setData: () => {},
    shiftKey,
  };
}

function _fireImageDragEvent(
  target: Element,
  type: string,
  files: File[],
  shiftKey: boolean,
) {
  const dt = _imageDataTransfer(files, shiftKey);
  const event = new Event(type, { bubbles: true, cancelable: true });
  Object.defineProperty(event, "dataTransfer", { value: dt });
  Object.defineProperty(event, "shiftKey", { value: shiftKey });
  target.dispatchEvent(event);
}

describe("ChatView image attachments (PR 11)", () => {
  it("paste of an image triggers uploadAttachment with persist=false", async () => {
    vi.mocked(Attachments.upload).mockResolvedValue({
      id: "img-1",
      filename: "Image.png",
      size_bytes: 4,
      persist: false,
      extract_chars: 0,
    });
    await renderWithLoadedConversation();

    const textarea = screen.getByTestId("chat-input") as HTMLTextAreaElement;
    const img = _makeImageFile("Image.png");
    const pasteEvent = new Event("paste", { bubbles: true, cancelable: true });
    Object.defineProperty(pasteEvent, "clipboardData", {
      value: {
        items: [
          { kind: "file", type: img.type, getAsFile: () => img },
        ],
      },
    });
    textarea.dispatchEvent(pasteEvent);

    await waitFor(() => {
      expect(Attachments.upload).toHaveBeenCalledTimes(1);
    });
    const call = vi.mocked(Attachments.upload).mock.calls[0];
    expect(call[0]).toBe("conv-A");
    expect((call[1] as File).name).toBe("Image.png");
    expect(call[2]).toBe(false);
  });

  it("drop of an image with Shift held still uses persist=false", async () => {
    vi.mocked(Attachments.upload).mockResolvedValue({
      id: "img-2",
      filename: "snap.png",
      size_bytes: 4,
      persist: false,
      extract_chars: 0,
    });
    await renderWithLoadedConversation();

    const target = screen.getByTestId("chat-drop-target");
    const img = _makeImageFile("snap.png");
    _fireImageDragEvent(target, "dragenter", [img], true);
    _fireImageDragEvent(target, "drop", [img], true);

    await waitFor(() => {
      expect(Attachments.upload).toHaveBeenCalledTimes(1);
    });
    // Shift is ignored for images — the override forces persist=false.
    expect(vi.mocked(Attachments.upload).mock.calls[0][2]).toBe(false);
  });

  it("dragging an image shows the 'always ephemeral' note in the overlay", async () => {
    await renderWithLoadedConversation();

    const target = screen.getByTestId("chat-drop-target");
    _fireImageDragEvent(target, "dragenter", [_makeImageFile()], false);

    await waitFor(() => {
      expect(screen.getByTestId("chat-drop-overlay")).toBeTruthy();
    });
    const hint = screen.getByTestId("chat-drop-overlay-hint");
    expect(hint.textContent).toMatch(/ephemeral/i);
  });

  it("file picker opens with accept=image/* and uploads with persist=false", async () => {
    vi.mocked(Attachments.upload).mockResolvedValue({
      id: "img-3",
      filename: "pick.png",
      size_bytes: 4,
      persist: false,
      extract_chars: 0,
    });
    await renderWithLoadedConversation();

    const picker = screen.getByTestId("chat-image-input") as HTMLInputElement;
    expect(picker.accept).toContain("image/");

    // Simulate the file picker selecting an image.
    const img = _makeImageFile("pick.png");
    Object.defineProperty(picker, "files", {
      value: {
        length: 1,
        0: img,
        item: (i: number) => (i === 0 ? img : null),
      },
      configurable: true,
    });
    picker.dispatchEvent(new Event("change", { bubbles: true }));

    await waitFor(() => {
      expect(Attachments.upload).toHaveBeenCalledTimes(1);
    });
    expect(vi.mocked(Attachments.upload).mock.calls[0][2]).toBe(false);
  });

  it("renders an image chip with a thumbnail and X to remove", async () => {
    const seeded = {
      id: "img-4",
      conversation_id: "conv-A",
      filename: "thumb.png",
      mime_type: "image/png",
      size_bytes: 12,
      persist: false,
      rag_doc_id: null,
      created_at: "2026-05-01T10:00:00Z",
    };
    vi.mocked(Attachments.list).mockResolvedValue([seeded]);
    vi.mocked(Attachments.fetchBlob).mockResolvedValue(
      new Blob([new Uint8Array([1, 2, 3])], { type: "image/png" }),
    );
    vi.mocked(Attachments.delete).mockResolvedValue({ ok: true });
    await renderWithLoadedConversation();

    await waitFor(() => {
      expect(screen.getByTestId("chat-attachment-chip-img-4")).toBeTruthy();
    });
    // Wait for the async blob fetch + thumbnail render.
    await waitFor(() => {
      expect(screen.getByTestId("chat-attachment-thumb-img-4")).toBeTruthy();
    });

    await userEvent.click(screen.getByTestId("chat-attachment-remove-img-4"));
    await waitFor(() => {
      expect(Attachments.delete).toHaveBeenCalledWith("img-4");
    });
  });
});

// ── Cross-conversation search (PR 13) ───────────────────────────────────────

describe("ChatView conversation search", () => {
  const FAKE_RESULTS = [
    {
      message_id: "msg-A",
      conversation_id: "conv-A",
      conversation_title: "Alpha chat",
      role: "user" as const,
      snippet: "How do I make <mark>fluffy</mark> pancakes?",
      created_at: "2026-05-01T10:00:00Z",
      rank: -2.5,
    },
    {
      message_id: "msg-B",
      conversation_id: "conv-B",
      conversation_title: "Beta chat",
      role: "assistant" as const,
      snippet: "<mark>Fluffy</mark> means light and airy.",
      created_at: "2026-05-02T10:00:00Z",
      rank: -1.0,
    },
  ];

  it("empty search input renders the conversation list", async () => {
    await renderWithLoadedConversation();
    // The conversation title appears in both the list and the header,
    // so just assert the search panel is absent and no fetch happened.
    expect(screen.queryByTestId("chat-search-results")).toBeNull();
    expect(screen.queryByTestId("chat-search-empty")).toBeNull();
    expect(Chat.searchConversations).not.toHaveBeenCalled();
  });

  it("typing a query triggers searchConversations after debounce", async () => {
    vi.mocked(Chat.searchConversations).mockResolvedValue(FAKE_RESULTS);
    await renderWithLoadedConversation();

    const search = screen.getByTestId("chat-search-input") as HTMLInputElement;
    await userEvent.type(search, "fluffy");

    await waitFor(() => {
      expect(Chat.searchConversations).toHaveBeenCalled();
    });
    // userEvent.type fires per keystroke; debounce should collapse those
    // to a single call with the full string.
    const calls = vi.mocked(Chat.searchConversations).mock.calls;
    expect(calls[calls.length - 1][0]).toBe("fluffy");
  });

  it("results render with snippet markup and replace the conversation list", async () => {
    vi.mocked(Chat.searchConversations).mockResolvedValue(FAKE_RESULTS);
    await renderWithLoadedConversation();

    const search = screen.getByTestId("chat-search-input") as HTMLInputElement;
    await userEvent.type(search, "fluffy");

    await waitFor(() => {
      expect(screen.getByTestId("chat-search-results")).toBeTruthy();
    });
    // The results panel replaces the conversation list inside the
    // search-results container — grab it and confirm the list buttons
    // (which would have data-testid undefined) are not children of it.
    const panel = screen.getByTestId("chat-search-results");
    // Both result rows render with their snippets inside the panel.
    expect(panel.querySelector('[data-testid="chat-search-result-msg-A"]')).toBeTruthy();
    expect(panel.querySelector('[data-testid="chat-search-result-msg-B"]')).toBeTruthy();

    // The <mark> token is rendered as a real <mark> element, not as text.
    const marks = panel.querySelectorAll("mark");
    expect(marks.length).toBeGreaterThanOrEqual(2);
    expect(marks[0].textContent).toBe("fluffy");
  });

  it("clicking a result switches activeConversationId and clears the input", async () => {
    vi.mocked(Chat.searchConversations).mockResolvedValue(FAKE_RESULTS);
    // After switching, ChatView re-fetches messages for the new id; let
    // that resolve to an empty list so the test doesn't blow up.
    vi.mocked(Chat.messages).mockImplementation(async (id: string) => {
      if (id === "conv-B") return [];
      return FAKE_MESSAGES;
    });
    await renderWithLoadedConversation();

    const search = screen.getByTestId("chat-search-input") as HTMLInputElement;
    await userEvent.type(search, "fluffy");

    await waitFor(() => {
      expect(screen.getByTestId("chat-search-result-msg-B")).toBeTruthy();
    });

    await userEvent.click(screen.getByTestId("chat-search-result-msg-B"));

    // Switching activeId triggers a fresh Chat.messages call for the new id.
    await waitFor(() => {
      expect(Chat.messages).toHaveBeenCalledWith("conv-B");
    });
    // The input is cleared, results panel disappears.
    expect(search.value).toBe("");
    await waitFor(() => {
      expect(screen.queryByTestId("chat-search-results")).toBeNull();
    });
  });

  it("Esc clears the search input", async () => {
    vi.mocked(Chat.searchConversations).mockResolvedValue(FAKE_RESULTS);
    await renderWithLoadedConversation();

    const search = screen.getByTestId("chat-search-input") as HTMLInputElement;
    await userEvent.type(search, "fluffy");
    expect(search.value).toBe("fluffy");

    await userEvent.type(search, "{Escape}");

    expect(search.value).toBe("");
  });
});

// ── PR 17: voice input ──────────────────────────────────────────────────────

// Drive a fake MediaRecorder so the mic-button path can run end-to-end
// without jsdom shipping the real one. Each instance captures the last
// constructed recorder so tests can fire ondataavailable / onstop manually.
class _FakeMediaRecorder {
  static instances: _FakeMediaRecorder[] = [];
  static isTypeSupported = vi.fn(() => true);
  state: "inactive" | "recording" | "paused" = "inactive";
  ondataavailable: ((e: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  onerror: ((e: unknown) => void) | null = null;

  constructor(_stream: MediaStream, _opts?: { mimeType?: string }) {
    _FakeMediaRecorder.instances.push(this);
  }

  start() {
    this.state = "recording";
  }

  stop() {
    this.state = "inactive";
    // Fire the data + stop callbacks asynchronously so the test sees the
    // same event ordering the real DOM API exposes.
    setTimeout(() => {
      this.ondataavailable?.({
        data: new Blob([new Uint8Array([1, 2, 3, 4])], { type: "audio/webm" }),
      });
      this.onstop?.();
    }, 0);
  }
}

function _stubMediaRecorder() {
  Object.defineProperty(globalThis, "MediaRecorder", {
    configurable: true,
    writable: true,
    value: _FakeMediaRecorder,
  });
}

function _stubGetUserMedia() {
  const fakeStream = {
    getTracks: () => [{ stop: vi.fn() }],
  } as unknown as MediaStream;
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: {
      getUserMedia: vi.fn().mockResolvedValue(fakeStream),
    },
  });
}

describe("ChatView voice input (PR 17)", () => {
  beforeEach(() => {
    _FakeMediaRecorder.instances = [];
    _stubMediaRecorder();
    _stubGetUserMedia();
  });

  it("does not render the mic button when voice_input_enabled is false", async () => {
    vi.mocked(Settings.get).mockResolvedValue({
      power_mode_enabled: false,
      voice_input_enabled: false,
      voice_output_enabled: false,
    } as never);
    await renderWithLoadedConversation();
    expect(screen.queryByTestId("chat-mic-button")).toBeNull();
  });

  it("renders the mic button when voice_input_enabled is true", async () => {
    vi.mocked(Settings.get).mockResolvedValue({
      power_mode_enabled: false,
      voice_input_enabled: true,
      voice_output_enabled: false,
    } as never);
    await renderWithLoadedConversation();
    await waitFor(() => {
      expect(screen.getByTestId("chat-mic-button")).toBeTruthy();
    });
  });

  it("clicking mic toggles isRecording and starts MediaRecorder", async () => {
    vi.mocked(Settings.get).mockResolvedValue({
      power_mode_enabled: false,
      voice_input_enabled: true,
      voice_output_enabled: false,
    } as never);
    await renderWithLoadedConversation();
    await waitFor(() => {
      expect(screen.getByTestId("chat-mic-button")).toBeTruthy();
    });

    await userEvent.click(screen.getByTestId("chat-mic-button"));

    await waitFor(() => {
      expect(useAppStore.getState().voiceRecording.isRecording).toBe(true);
    });
    expect(_FakeMediaRecorder.instances.length).toBeGreaterThan(0);

    // Recording indicator visible.
    expect(screen.getByTestId("chat-recording-indicator")).toBeTruthy();
  });

  it("stopping the recording calls Voice.transcribe and populates the input", async () => {
    vi.mocked(Settings.get).mockResolvedValue({
      power_mode_enabled: false,
      voice_input_enabled: true,
      voice_output_enabled: false,
    } as never);
    vi.mocked(Voice.transcribe).mockResolvedValue({ text: "hello world" });
    await renderWithLoadedConversation();
    await waitFor(() => {
      expect(screen.getByTestId("chat-mic-button")).toBeTruthy();
    });

    // Start
    await userEvent.click(screen.getByTestId("chat-mic-button"));
    await waitFor(() => {
      expect(useAppStore.getState().voiceRecording.isRecording).toBe(true);
    });

    // Stop — fires the fake recorder's ondataavailable + onstop on a
    // microtask, which kicks off Voice.transcribe.
    await userEvent.click(screen.getByTestId("chat-mic-button"));

    await waitFor(() => {
      expect(Voice.transcribe).toHaveBeenCalledTimes(1);
    });
    await waitFor(() => {
      const textarea = screen.getByTestId("chat-input") as HTMLTextAreaElement;
      expect(textarea.value).toContain("hello world");
    });
    // Recording state cleared.
    await waitFor(() => {
      expect(useAppStore.getState().voiceRecording.isRecording).toBe(false);
      expect(useAppStore.getState().voiceRecording.isTranscribing).toBe(false);
    });
  });
});

// ── PR 18: slash-command snippet picker ────────────────────────────────────

describe("ChatView slash command snippet picker", () => {
  const SNIPPET = {
    id: "snip-fluffy",
    title: "Fluffy",
    body: "Make it fluffy.",
    kind: "snippet" as const,
    tags: "tone",
    created_at: "2026-05-01T10:00:00Z",
    updated_at: "2026-05-01T10:00:00Z",
    use_count: 1,
  };
  const SYS_PROMPT = {
    id: "sys-formal",
    title: "Formal",
    body: "Be formal.",
    kind: "system_prompt" as const,
    tags: "tone",
    created_at: "2026-05-01T10:00:00Z",
    updated_at: "2026-05-01T10:00:00Z",
    use_count: 0,
  };

  it("typing '/' at start of input shows the snippet dropdown", async () => {
    vi.mocked(PromptTemplates.list).mockResolvedValue([SNIPPET, SYS_PROMPT]);
    await renderWithLoadedConversation();

    const textarea = screen.getByTestId("chat-input") as HTMLTextAreaElement;
    await userEvent.type(textarea, "/");

    await waitFor(() => {
      expect(screen.getByTestId("chat-slash-dropdown")).toBeTruthy();
    });
    // Snippet appears, system_prompt does not.
    await waitFor(() => {
      expect(screen.getByTestId("chat-slash-option-snip-fluffy")).toBeTruthy();
    });
    expect(screen.queryByTestId("chat-slash-option-sys-formal")).toBeNull();
  });

  it("selecting a snippet from the dropdown inserts its body and calls /use", async () => {
    vi.mocked(PromptTemplates.list).mockResolvedValue([SNIPPET]);
    vi.mocked(PromptTemplates.use).mockResolvedValue({
      ...SNIPPET,
      use_count: 2,
    });
    await renderWithLoadedConversation();

    const textarea = screen.getByTestId("chat-input") as HTMLTextAreaElement;
    await userEvent.type(textarea, "/");

    await waitFor(() => {
      expect(screen.getByTestId("chat-slash-option-snip-fluffy")).toBeTruthy();
    });

    // mousedown is what the option uses to fire the pick (so blur doesn't
    // race the click). userEvent.click fires mousedown + mouseup + click,
    // which exercises the same path.
    await userEvent.click(
      screen.getByTestId("chat-slash-option-snip-fluffy"),
    );

    await waitFor(() => {
      expect((screen.getByTestId("chat-input") as HTMLTextAreaElement).value)
        .toBe("Make it fluffy.");
    });
    await waitFor(() => {
      expect(PromptTemplates.use).toHaveBeenCalledWith("snip-fluffy");
    });
    // Dropdown closes after a pick.
    await waitFor(() => {
      expect(screen.queryByTestId("chat-slash-dropdown")).toBeNull();
    });
  });

  it("typing after the slash filters the visible snippets", async () => {
    const SNIPPET_B = {
      ...SNIPPET,
      id: "snip-other",
      title: "Other thing",
      tags: "misc",
    };
    vi.mocked(PromptTemplates.list).mockResolvedValue([SNIPPET, SNIPPET_B]);
    await renderWithLoadedConversation();

    const textarea = screen.getByTestId("chat-input") as HTMLTextAreaElement;
    await userEvent.type(textarea, "/fluf");

    await waitFor(() => {
      expect(screen.getByTestId("chat-slash-option-snip-fluffy")).toBeTruthy();
    });
    expect(screen.queryByTestId("chat-slash-option-snip-other")).toBeNull();
  });
});
