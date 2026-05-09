import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { MessageRenderer } from "./MessageRenderer";

// Mock the Voice API client so the speaker button can run end-to-end
// without hitting the network. Tests override the synthesize return value.
vi.mock("@/api/client", () => ({
  Voice: {
    synthesize: vi.fn(),
  },
}));

import { Voice } from "@/api/client";

// Mermaid (and its d3 dependency) reach for browser APIs that jsdom doesn't
// implement (XMLSerializer behaviors, getBBox, etc.). Mock the module so the
// component-side contract — call render(id, src), inject the returned
// SVG; on rejection show the fallback caption — is what we exercise.
vi.mock("mermaid", () => ({
  default: {
    initialize: vi.fn(),
    render: vi.fn(async (_id: string, src: string) => {
      if (src.includes("__BAD__")) {
        throw new Error("parse error");
      }
      return { svg: `<svg data-testid="mermaid-svg"><text>${src}</text></svg>` };
    }),
  },
}));

// jsdom doesn't ship navigator.clipboard. Stub it before each test so the
// CopyButton's writeText call succeeds and we can assert on the args.
function stubClipboard() {
  const writeText = vi.fn().mockResolvedValue(undefined);
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText },
  });
  return writeText;
}

beforeEach(() => {
  stubClipboard();
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("MessageRenderer", () => {
  it("renders user message as plain text without parsing markdown", () => {
    const { container } = render(
      <MessageRenderer content="**not bold** plain text" role="user" />,
    );
    // Plain text path keeps the asterisks literal — no <strong> emitted.
    expect(container.querySelector("strong")).toBeNull();
    expect(container.textContent).toBe("**not bold** plain text");
  });

  it("preserves newlines for user messages via whitespace-pre-wrap", () => {
    const { container } = render(
      <MessageRenderer content={"line one\nline two"} role="user" />,
    );
    const span = container.querySelector("span");
    expect(span?.className).toContain("whitespace-pre-wrap");
    expect(span?.textContent).toBe("line one\nline two");
  });

  it("renders markdown bold and italic in assistant messages", () => {
    const { container } = render(
      <MessageRenderer content="**bold** and *italic*" role="assistant" />,
    );
    expect(container.querySelector("strong")?.textContent).toBe("bold");
    expect(container.querySelector("em")?.textContent).toBe("italic");
  });

  it("renders fenced code block with a copy button", () => {
    render(
      <MessageRenderer
        content={"```js\nconst x = 1;\n```"}
        role="assistant"
      />,
    );
    expect(
      screen.getByRole("button", { name: /copy code/i }),
    ).toBeTruthy();
  });

  it("clicking the copy button writes the code content to the clipboard", async () => {
    const writeText = stubClipboard();
    render(
      <MessageRenderer
        content={"```python\nprint('hi')\n```"}
        role="assistant"
      />,
    );
    await userEvent.click(
      screen.getByRole("button", { name: /copy code/i }),
    );
    expect(writeText).toHaveBeenCalledTimes(1);
    expect(writeText.mock.calls[0]?.[0]).toContain("print('hi')");
    // Button label flips to "Copied" for the 1.5s confirmation window.
    expect(
      screen.getByRole("button", { name: /copied/i }),
    ).toBeTruthy();
  });

  it("renders a GFM table as a <table> element", () => {
    const md = [
      "| col1 | col2 |",
      "| ---- | ---- |",
      "| a    | b    |",
    ].join("\n");
    const { container } = render(
      <MessageRenderer content={md} role="assistant" />,
    );
    const table = container.querySelector("table");
    expect(table).not.toBeNull();
    expect(container.querySelectorAll("th").length).toBe(2);
    expect(container.querySelectorAll("td").length).toBe(2);
  });

  it("does not throw on incomplete fenced code mid-stream", () => {
    expect(() =>
      render(
        <MessageRenderer
          content={"here is some code:\n```js\nconst x ="}
          role="assistant"
        />,
      ),
    ).not.toThrow();
  });

  it("renders inline code without a copy button", () => {
    render(
      <MessageRenderer
        content="use `const x = 1` inline"
        role="assistant"
      />,
    );
    expect(screen.queryByRole("button", { name: /copy/i })).toBeNull();
  });

  it("renders inline math via KaTeX", () => {
    const { container } = render(
      <MessageRenderer content={"$E = mc^2$"} role="assistant" />,
    );
    // rehype-katex emits a span.katex wrapping span.katex-html for
    // browser rendering and span.katex-mathml for accessibility.
    const katex = container.querySelector(".katex");
    expect(katex).not.toBeNull();
    expect(container.querySelector(".katex-html")).not.toBeNull();
  });

  it("renders display math via KaTeX with the display class", () => {
    // remark-math: a `$$..$$` block needs to stand on its own (no
    // surrounding inline text on the same paragraph) to be treated as a
    // display block rather than inline math.
    const { container } = render(
      <MessageRenderer
        content={"before\n\n$$\n\\sum_{i=1}^n i\n$$\n\nafter"}
        role="assistant"
      />,
    );
    expect(container.querySelector(".katex-display")).not.toBeNull();
  });

  it("does not throw on incomplete inline math mid-stream", () => {
    expect(() =>
      render(
        <MessageRenderer
          content={"here is math: $E = mc"}
          role="assistant"
        />,
      ),
    ).not.toThrow();
  });

  it("renders a mermaid fenced block as an SVG", async () => {
    const { container } = render(
      <MessageRenderer
        content={"```mermaid\ngraph TD; A-->B;\n```"}
        role="assistant"
      />,
    );
    await waitFor(() => {
      expect(container.querySelector("svg")).not.toBeNull();
    });
  });

  it("falls back to a code block with caption when mermaid render fails", async () => {
    const { container } = render(
      <MessageRenderer
        content={"```mermaid\n__BAD__ syntax\n```"}
        role="assistant"
      />,
    );
    await waitFor(() => {
      expect(container.textContent).toContain("Diagram render failed");
    });
    // Fallback: SVG should NOT be present, but the source should be
    // visible inside a <pre><code>.
    expect(container.querySelector("svg")).toBeNull();
    expect(container.querySelector("pre code")?.textContent).toContain(
      "__BAD__",
    );
  });

  it("renders an unclosed mermaid fence as a plain code block (no throw)", () => {
    expect(() =>
      render(
        <MessageRenderer
          content={"```mermaid\ngraph TD; A--"}
          role="assistant"
        />,
      ),
    ).not.toThrow();
    // Whether the parser closes or not, we never blow up; the source is
    // visible to the user.
    expect(screen.getByText(/graph TD/)).toBeTruthy();
  });

  it("still highlights non-mermaid fenced code (PR 4 regression)", () => {
    const { container } = render(
      <MessageRenderer
        content={"```js\nconst x = 1;\n```"}
        role="assistant"
      />,
    );
    // rehype-highlight stamps `hljs language-…` on the <code> element.
    const code = container.querySelector("pre code");
    expect(code?.className).toMatch(/hljs|language-js/);
  });
});

// ── PR 17: speaker button ───────────────────────────────────────────────────

describe("MessageRenderer speaker button (PR 17)", () => {
  beforeEach(() => {
    vi.mocked(Voice.synthesize).mockReset();
    // Stub Audio so play() doesn't throw under jsdom.
    Object.defineProperty(globalThis, "Audio", {
      configurable: true,
      writable: true,
      value: class {
        src = "";
        currentTime = 0;
        _listeners: Record<string, Array<() => void>> = {};
        addEventListener(event: string, fn: () => void) {
          (this._listeners[event] ||= []).push(fn);
        }
        play() {
          return Promise.resolve();
        }
        pause() {}
      },
    });
    if (typeof URL.createObjectURL !== "function") {
      Object.defineProperty(URL, "createObjectURL", {
        configurable: true,
        value: vi.fn(() => "blob:fake-tts"),
      });
    }
    if (typeof URL.revokeObjectURL !== "function") {
      Object.defineProperty(URL, "revokeObjectURL", {
        configurable: true,
        value: vi.fn(),
      });
    }
  });

  it("does not render the speaker button when voiceOutputEnabled is false", () => {
    render(
      <MessageRenderer
        content="Hello there."
        role="assistant"
        voiceOutputEnabled={false}
      />,
    );
    expect(screen.queryByTestId("message-speaker-button")).toBeNull();
  });

  it("renders the speaker button only on assistant messages with voiceOutputEnabled", () => {
    render(
      <MessageRenderer
        content="Hello there."
        role="assistant"
        voiceOutputEnabled={true}
      />,
    );
    expect(screen.getByTestId("message-speaker-button")).toBeTruthy();
  });

  it("does not render the speaker button on user messages", () => {
    render(
      <MessageRenderer
        content="Hi back"
        role="user"
        voiceOutputEnabled={true}
      />,
    );
    expect(screen.queryByTestId("message-speaker-button")).toBeNull();
  });

  it("clicking the speaker calls Voice.synthesize with the message text", async () => {
    vi.mocked(Voice.synthesize).mockResolvedValue(
      new Blob([new Uint8Array([1, 2, 3])], { type: "audio/wav" }),
    );
    render(
      <MessageRenderer
        content="Hello world"
        role="assistant"
        voiceOutputEnabled={true}
      />,
    );
    await userEvent.click(screen.getByTestId("message-speaker-button"));
    await waitFor(() => {
      expect(Voice.synthesize).toHaveBeenCalledTimes(1);
    });
    const arg = vi.mocked(Voice.synthesize).mock.calls[0][0];
    expect(arg).toContain("Hello world");
  });

  it("strips Markdown before sending the text to synthesis", async () => {
    vi.mocked(Voice.synthesize).mockResolvedValue(
      new Blob([new Uint8Array([1, 2, 3])], { type: "audio/wav" }),
    );
    render(
      <MessageRenderer
        content={"This **is** a `code` test"}
        role="assistant"
        voiceOutputEnabled={true}
      />,
    );
    await userEvent.click(screen.getByTestId("message-speaker-button"));
    await waitFor(() => {
      expect(Voice.synthesize).toHaveBeenCalled();
    });
    const arg = vi.mocked(Voice.synthesize).mock.calls[0][0];
    expect(arg).not.toContain("**");
    expect(arg).not.toContain("`");
  });
});
