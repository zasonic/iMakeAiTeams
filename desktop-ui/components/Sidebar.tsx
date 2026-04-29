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
  { id: "rag", label: "Documents", hint: "Index files and folders" },
  { id: "memory", label: "Memory", hint: "Search session facts" },
  { id: "prompts", label: "Prompts", hint: "Manage system prompts", studioOnly: true },
  { id: "mcp", label: "MCP", hint: "Tool servers", studioOnly: true },
  { id: "security", label: "Security", hint: "Firewall + scan log", studioOnly: true },
  { id: "settings", label: "Settings", hint: "API keys, models, routing" },
  { id: "diagnostics", label: "Diagnostics", hint: "Health + error logs", studioOnly: true },
];

export function Sidebar() {
  const active = useAppStore((s) => s.activeView);
  const studio = useAppStore((s) => s.studioMode);
  const setActive = useAppStore((s) => s.setActiveView);
  const setStudio = useAppStore((s) => s.setStudioMode);

  const visible = NAV.filter((n) => studio || !n.studioOnly);

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
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => setActive(item.id)}
              className={`w-full text-left px-4 py-2 text-sm flex flex-col rounded-md mx-2 my-0.5 transition ${
                isActive
                  ? "bg-accent/10 text-ink"
                  : "text-ink-dim hover:bg-bg-2 hover:text-ink"
              }`}
            >
              <span className="font-medium">{item.label}</span>
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
