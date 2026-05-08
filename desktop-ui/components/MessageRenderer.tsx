// desktop-ui/components/MessageRenderer.tsx — render chat messages.
//
// User messages stay as plain text with whitespace-pre-wrap (existing
// behavior). Assistant messages flow through react-markdown with
// remark-gfm + rehype-highlight so tables, task lists, code fences, etc.
// render as expected. Code fences get a copy button; links open in the
// system browser via the existing IPC bridge when available.

import {
  useCallback,
  useEffect,
  useState,
  type AnchorHTMLAttributes,
  type HTMLAttributes,
  type MouseEvent,
  type ReactNode,
} from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";

import "highlight.js/styles/github-dark.css";

interface MessageRendererProps {
  content: string;
  role: "user" | "assistant" | "system";
}

export function MessageRenderer({ content, role }: MessageRendererProps) {
  if (role !== "assistant") {
    return <span className="whitespace-pre-wrap">{content}</span>;
  }
  return (
    <div className="markdown-body whitespace-normal">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[[rehypeHighlight, { detect: true, ignoreMissing: true }]]}
        components={MARKDOWN_COMPONENTS}
        skipHtml
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

const MARKDOWN_COMPONENTS: Components = {
  pre: PreBlock,
  code: InlineOrFencedCode,
  a: ExternalLink,
  table: ({ children, ...rest }) => (
    <div className="overflow-x-auto my-2">
      <table
        {...rest}
        className="border-collapse border border-line text-xs"
      >
        {children}
      </table>
    </div>
  ),
  th: ({ children, ...rest }) => (
    <th
      {...rest}
      className="border border-line bg-bg-2 px-2 py-1 text-left font-semibold"
    >
      {children}
    </th>
  ),
  td: ({ children, ...rest }) => (
    <td {...rest} className="border border-line px-2 py-1 align-top">
      {children}
    </td>
  ),
  ul: ({ children, ...rest }) => (
    <ul {...rest} className="list-disc pl-5 my-1.5 space-y-0.5">
      {children}
    </ul>
  ),
  ol: ({ children, ...rest }) => (
    <ol {...rest} className="list-decimal pl-5 my-1.5 space-y-0.5">
      {children}
    </ol>
  ),
  blockquote: ({ children, ...rest }) => (
    <blockquote
      {...rest}
      className="border-l-2 border-accent/50 pl-3 my-2 text-ink-dim"
    >
      {children}
    </blockquote>
  ),
  h1: ({ children, ...rest }) => (
    <h1 {...rest} className="text-base font-semibold mt-3 mb-1.5">
      {children}
    </h1>
  ),
  h2: ({ children, ...rest }) => (
    <h2 {...rest} className="text-sm font-semibold mt-3 mb-1.5">
      {children}
    </h2>
  ),
  h3: ({ children, ...rest }) => (
    <h3 {...rest} className="text-sm font-semibold mt-2 mb-1">
      {children}
    </h3>
  ),
  p: ({ children, ...rest }) => (
    <p {...rest} className="my-1.5 leading-relaxed">
      {children}
    </p>
  ),
};

// ── Code blocks ───────────────────────────────────────────────────────────────
//
// react-markdown wraps fenced code in <pre><code class="language-foo">…</code>.
// We intercept <pre> so the copy button can position itself relative to the
// block and still reach the raw text content. Inline code (no <pre> ancestor,
// no language class) is rendered with a subtle background and no button.

function PreBlock({ children }: HTMLAttributes<HTMLPreElement>) {
  const codeText = extractCodeText(children);
  return (
    <div className="relative group my-2">
      <pre className="rounded-md bg-bg-1 border border-line p-3 overflow-x-auto text-xs font-mono">
        {children}
      </pre>
      <CopyButton text={codeText} />
    </div>
  );
}

function InlineOrFencedCode({
  className,
  children,
  ...rest
}: HTMLAttributes<HTMLElement>) {
  // Fenced code blocks always carry a `language-…` class from rehype-highlight
  // (or `hljs` if detection ran). Inline code has neither.
  const isFenced = typeof className === "string" && /(^|\s)(language-|hljs)/.test(className);
  if (isFenced) {
    return (
      <code {...rest} className={className}>
        {children}
      </code>
    );
  }
  return (
    <code
      {...rest}
      className="rounded bg-bg-1 border border-line px-1 py-0.5 text-[0.85em] font-mono"
    >
      {children}
    </code>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return;
    const id = window.setTimeout(() => setCopied(false), 1500);
    return () => window.clearTimeout(id);
  }, [copied]);
  const onCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
    } catch {
      // Clipboard API may be unavailable (e.g. insecure context). Silently
      // ignore — the user can still select the text manually.
    }
  }, [text]);
  return (
    <button
      type="button"
      onClick={onCopy}
      aria-label={copied ? "Copied" : "Copy code"}
      className="absolute top-1.5 right-1.5 rounded border border-line bg-bg-2 px-2 py-0.5 text-[11px] text-ink-dim opacity-0 transition group-hover:opacity-100 hover:text-ink hover:bg-bg-3 focus:opacity-100"
    >
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

function extractCodeText(node: ReactNode): string {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string") return node;
  if (typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(extractCodeText).join("");
  if (typeof node === "object" && "props" in node) {
    const props = (node as { props?: { children?: ReactNode } }).props;
    return extractCodeText(props?.children);
  }
  return "";
}

// ── Links ─────────────────────────────────────────────────────────────────────

function ExternalLink({
  href,
  children,
  ...rest
}: AnchorHTMLAttributes<HTMLAnchorElement>) {
  const onClick = (e: MouseEvent<HTMLAnchorElement>) => {
    if (!href) return;
    // If the IPC bridge is present, route through shell.openExternal so links
    // open in the user's default browser instead of inside the Electron
    // window. Fallback: let the anchor behave normally with the safe rel set.
    const api = (window as Window & { electronAPI?: { openExternal?: (url: string) => Promise<void> } }).electronAPI;
    if (api?.openExternal) {
      e.preventDefault();
      api.openExternal(href).catch(() => {});
    }
  };
  return (
    <a
      {...rest}
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      onClick={onClick}
      className="text-accent hover:underline"
    >
      {children}
    </a>
  );
}
