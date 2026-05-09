import { useAppStore, type ActiveView } from "@/stores/appStore";

interface NavItem {
  id: ActiveView;
  label: string;
  hint: string;
  studioOnly?: boolean;
}

const NAV: NavItem[] = [
  { id: "chat", label: "Chat", hint: "Talk to your team" },
  { id: "agents", label: "Agents", hint: "Define agents and teams" },
  { id: "models", label: "Models", hint: "Browse and switch local models" },
  { id: "rag", label: "Documents", hint: "Index files and folders" },
  { id: "memory", label: "Memory", hint: "Search session facts" },
  { id: "memory_review", label: "Memory Review", hint: "Approve memory writes" },
  { id: "escalations", label: "Pending Reviews", hint: "Approve paused actions" },
  { id: "prompts", label: "Prompts", hint: "Manage system prompts", studioOnly: true },
  { id: "saved_prompts", label: "Saved prompts", hint: "Snippets and system prompt templates" },
  { id: "mcp", label: "MCP", hint: "Tool servers", studioOnly: true },
  { id: "security", label: "Security", hint: "Firewall + scan log", studioOnly: true },
  { id: "safety", label: "Safety", hint: "Escalations, gate, canary, voting" },
  { id: "usage", label: "Usage", hint: "Token consumption and cost" },
  { id: "settings", label: "Settings", hint: "API keys, models, routing" },
  { id: "diagnostics", label: "Diagnostics", hint: "Health + error logs", studioOnly: true },
];

export function Sidebar() {
  const active = useAppStore((s) => s.activeView);
  const studio = useAppStore((s) => s.studioMode);
  const setActive = useAppStore((s) => s.setActiveView);
  const setStudio = useAppStore((s) => s.setStudioMode);
  const pendingCount = useAppStore((s) => s.pendingEscalations.length);
  const memoryReviewCount = useAppStore((s) => s.pendingMemoryWrites.length);

  const visible = NAV.filter((n) => studio || !n.studioOnly);

  const badgeCount = (id: ActiveView): number => {
    if (id === "escalations") return pendingCount;
    if (id === "memory_review") return memoryReviewCount;
    return 0;
  };

  return (
    <aside className="flex flex-col w-56 min-w-56 border-r border-line bg-bg-1">
      <div className="px-4 py-4 border-b border-line">
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-accent to-claude flex items-center justify-center text-white font-bold">
            ai
          </div>
          <div className="leading-tight">
            <div className="text-sm font-semibold">iMakeAiTeams</div>
            <div className="text-[10px] uppercase tracking-wide text-ink-faint">
              Local-first
            </div>
          </div>
        </div>
      </div>

      <nav className="flex-1 overflow-y-auto py-2">
        {visible.map((item) => {
          const isActive = active === item.id;
          const count = badgeCount(item.id);
          const showBadge = count > 0;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => setActive(item.id)}
              aria-label={item.label}
              aria-current={isActive ? "page" : undefined}
              className={`w-full text-left px-4 py-2 text-sm flex flex-col rounded-md mx-2 my-0.5 transition ${
                isActive
                  ? "bg-accent/10 text-ink"
                  : "text-ink-dim hover:bg-bg-2 hover:text-ink"
              }`}
            >
              <span className="font-medium flex items-center gap-2">
                {item.label}
                {showBadge && (
                  <span className="inline-flex items-center justify-center min-w-[18px] h-[18px] px-1 text-[10px] rounded-full bg-warn/20 text-warn">
                    {count}
                  </span>
                )}
              </span>
              <span className="text-[11px] text-ink-faint">{item.hint}</span>
            </button>
          );
        })}
      </nav>

      <div className="border-t border-line px-4 py-3 flex items-center justify-between text-xs text-ink-dim">
        <span>Studio mode</span>
        <button
          type="button"
          onClick={() => setStudio(!studio)}
          className={`h-5 w-9 rounded-full transition relative ${
            studio ? "bg-accent" : "bg-bg-3"
          }`}
          aria-pressed={studio}
          aria-label="Toggle studio mode"
        >
          <span
            className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-all ${
              studio ? "left-4" : "left-0.5"
            }`}
          />
        </button>
      </div>
    </aside>
  );
}
