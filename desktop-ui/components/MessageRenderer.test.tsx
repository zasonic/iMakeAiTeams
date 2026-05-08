import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { MessageRenderer } from "./MessageRenderer";

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
});
