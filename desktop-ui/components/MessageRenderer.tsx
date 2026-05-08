// desktop-ui/components/MessageRenderer.tsx — render chat messages.
//
// User messages stay as plain text with whitespace-pre-wrap (existing
// behavior). Assistant messages flow through react-markdown with
// remark-gfm + remark-math + rehype-highlight + rehype-katex so tables,
// task lists, code fences, math, etc. render as expected. Mermaid fenced
// code blocks are intercepted and rendered as SVG diagrams. Code fences
// get a copy button; links open in the system browser via the existing
// IPC bridge when available.

import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type AnchorHTMLAttributes,
  type HTMLAttributes,
  type MouseEvent,
  type ReactNode,
} from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

import "highlight.js/styles/github-dark.css";
import "katex/dist/katex.min.css";

interface MessageRendererProps {
  content: string;
  role: "user" | "assistant" | "system";
}

// Mermaid is heavy (~700 kB minified plus per-diagram chunks). We lazy-load
// it on first use so chats without diagrams pay nothing. The app currently
// ships a single dark theme (see tailwind.config.js — darkMode is "class"
// but no runtime toggle exists). When a theme switcher lands, re-initialize
// on theme change. TODO(theme-toggle): wire mermaid theme to the app theme
// when a toggle is introduced.
type MermaidModule = typeof import("mermaid").default;
let mermaidPromise: Promise<MermaidModule> | null = null;
function loadMermaid(): Promise<MermaidModule> {
  if (!mermaidPromise) {
    mermaidPromise = import("mermaid").then((mod) => {
      mod.default.initialize({
        startOnLoad: false,
        theme: "dark",
        securityLevel: "strict",
        fontFamily: "inherit",
      });
      return mod.default;
    });
  }
  return mermaidPromise;
}

export function MessageRenderer({ content, role }: MessageRendererProps) {
  if (role !== "assistant") {
    return <span className="whitespace-pre-wrap">{content}</span>;
  }
  return (
    <div className="markdown-body whitespace-normal">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[
          [rehypeHighlight, { detect: true, ignoreMissing: true }],
          // Defensive: rehype-katex catches syntax errors by default and
          // renders the raw expression with an error color, so a
          // mid-stream half-written `$$\sum_{i=1` (once the closing $$
          // arrives) won't throw out of the render. `strict: "ignore"`
          // silences warnings for unknown commands.
          [rehypeKatex, { throwOnError: false, strict: "ignore" }],
        ]}
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
//
// Fences tagged ```mermaid skip the normal code-block path entirely and go
// through MermaidDiagram, which lazily turns the source into an SVG.

function PreBlock({ children }: HTMLAttributes<HTMLPreElement>) {
  const codeText = extractCodeText(children);
  const lang = extractCodeLang(children);
  if (lang === "mermaid") {
    return <MermaidDiagram source={codeText} />;
  }
  return (
    <div className="relative group my-2">
      <pre className="rounded-md bg-bg-1 border border-line p-3 overflow-x-auto text-xs font-mono">
        {children}
      </pre>
      <CopyButton text={codeText} />
    </div>
  );
}

// ── Mermaid diagrams ──────────────────────────────────────────────────────────
//
// Mermaid render is async and can throw on malformed syntax. While the chat
// is streaming, the source updates in place and most intermediate states are
// invalid — we treat a render failure as "not yet renderable" and show the
// raw source as a plain code block. When a successful render arrives, we
// swap in the SVG.

function MermaidDiagram({ source }: { source: string }) {
  const [svg, setSvg] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  const reactId = useId();
  // mermaid requires a DOM-id-safe identifier for its render container.
  const renderId = `mermaid-${reactId.replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const latestSource = useRef(source);
  latestSource.current = source;

  useEffect(() => {
    let cancelled = false;
    const trimmed = source.trim();
    if (trimmed.length === 0) {
      setSvg(null);
      setFailed(false);
      return () => {
        cancelled = true;
      };
    }
    (async () => {
      try {
        const mermaid = await loadMermaid();
        const { svg: rendered } = await mermaid.render(renderId, trimmed);
        if (cancelled || latestSource.current !== source) return;
        setSvg(rendered);
        setFailed(false);
      } catch {
        if (cancelled || latestSource.current !== source) return;
        setSvg(null);
        setFailed(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [source, renderId]);

  if (svg != null) {
    return (
      <div
        className="my-2 overflow-x-auto rounded-md bg-bg-1 border border-line p-3"
        // eslint-disable-next-line react/no-danger -- mermaid output is
        // generated by a trusted library at strict security level.
        dangerouslySetInnerHTML={{ __html: svg }}
      />
    );
  }

  return (
    <div className="relative group my-2">
      <pre className="rounded-md bg-bg-1 border border-line p-3 overflow-x-auto text-xs font-mono">
        <code className="language-mermaid">{source}</code>
      </pre>
      <CopyButton text={source} />
      {failed && (
        <div className="mt-1 text-[11px] text-ink-faint italic">
          Diagram render failed
        </div>
      )}
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

// Inspect the <code> child of a <pre> to find its language tag, e.g. the
// "mermaid" in ```mermaid. react-markdown emits the tag as a "language-…"
// class on the <code> element.
function extractCodeLang(node: ReactNode): string | null {
  if (node == null || typeof node !== "object") return null;
  if (Array.isArray(node)) {
    for (const child of node) {
      const lang = extractCodeLang(child);
      if (lang) return lang;
    }
    return null;
  }
  if ("props" in node) {
    const props = (node as { props?: { className?: string } }).props;
    const className = props?.className;
    if (typeof className === "string") {
      const match = className.match(/language-(\S+)/);
      if (match) return match[1] ?? null;
    }
  }
  return null;
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
