"""
services/mcp_registry.py — Phase 2: in-memory MCP tool catalog.

The registry owns:
  - The set of discovered MCP servers under paths.mcp_servers_dir().
  - The per-server enable/disable list (loaded from settings).
  - On-demand resolution of tools relevant to a given task by skill-tag match.
  - Stable, prompt-token-bounded serialization of the resolved tool subset.

Hot reload: ``refresh_if_stale()`` checks the directory's mtime and re-scans
only when it changed. Any caller that takes a non-stale-aware shortcut should
be paired with a deliberate ``refresh()`` in the test.

Token-budget invariance: ``serialize_for_prompt(resolved_tools)`` only
serializes the resolved subset, so the prompt size scales with the resolved
set, not the catalog. This is the property tested by the Phase 2 plan.

Phase 2 explicitly does NOT execute MCP tools. The Anthropic-shaped tool
schemas are exposed for a future execution phase (Phase 5+).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional

from services.mcp_loader import MCPServer, ToolSchema, scan_servers

log = logging.getLogger("MyAIEnv.mcp_registry")

PROMPT_HEADING = "## Available tools (MCP)"
SETTING_DISABLED_KEY = "mcp_servers_disabled"


class MCPRegistry:
    """Discovery + resolution boundary for MCP tools."""

    def __init__(self, root: Path, settings):
        self._root = Path(root)
        self._settings = settings
        self._servers: list[MCPServer] = []
        self._last_scan_mtime: float = -1.0
        # Initial scan happens lazily on first use so import time stays small.

    # ── Discovery / hot-reload ───────────────────────────────────────────────

    def _root_mtime(self) -> float:
        try:
            return self._root.stat().st_mtime
        except OSError:
            return -1.0

    def refresh(self) -> None:
        """Force-rescan the servers directory."""
        self._servers = scan_servers(self._root)
        self._last_scan_mtime = self._root_mtime()
        log.info(
            "MCP registry rescanned: %d server(s), %d tool(s) total",
            len(self._servers), sum(s.tool_count() for s in self._servers),
        )

    def refresh_if_stale(self) -> bool:
        """Rescan only when the root directory's mtime changed. Returns True if a scan ran."""
        cur = self._root_mtime()
        if cur != self._last_scan_mtime:
            self.refresh()
            return True
        return False

    # ── Disable / enable ────────────────────────────────────────────────────

    def _disabled_set(self) -> set[str]:
        raw = self._settings.get(SETTING_DISABLED_KEY, []) or []
        if isinstance(raw, list):
            return {str(x) for x in raw}
        return set()

    def is_enabled(self, server_id: str) -> bool:
        return server_id not in self._disabled_set()

    def set_enabled(self, server_id: str, enabled: bool) -> None:
        disabled = self._disabled_set()
        if enabled:
            disabled.discard(server_id)
        else:
            disabled.add(server_id)
        self._settings.set(SETTING_DISABLED_KEY, sorted(disabled))

    # ── Public catalog views ────────────────────────────────────────────────

    def list_servers(self) -> list[MCPServer]:
        self.refresh_if_stale()
        return list(self._servers)

    def get_server(self, server_id: str) -> Optional[MCPServer]:
        self.refresh_if_stale()
        for s in self._servers:
            if s.server_id == server_id:
                return s
        return None

    def all_tools(self, *, include_disabled: bool = False) -> list[ToolSchema]:
        self.refresh_if_stale()
        if include_disabled:
            return [t for s in self._servers for t in s.tools]
        disabled = self._disabled_set()
        return [
            t for s in self._servers
            if s.server_id not in disabled
            for t in s.tools
        ]

    # ── Resolution ──────────────────────────────────────────────────────────

    def resolve_for_task(self, required_skills: Iterable[str],
                         required_scopes: Iterable[str] = ()) -> list[ToolSchema]:
        """
        Return tools whose ``skill_tags`` intersect ``required_skills``. If
        ``required_scopes`` is non-empty, only tools whose declared scopes are
        a superset of the requested scopes are returned. Tools with no
        ``skill_tags`` are excluded — callers wanting them should pass an
        explicit "general" skill tag and have those tools declare it.
        """
        skills_set = {str(s) for s in required_skills if str(s).strip()}
        scopes_set = {str(s) for s in required_scopes if str(s).strip()}
        if not skills_set:
            return []
        out: list[ToolSchema] = []
        for tool in self.all_tools():
            tags = set(tool.skill_tags)
            if not tags or tags.isdisjoint(skills_set):
                continue
            if scopes_set and not scopes_set.issubset(set(tool.scopes)):
                continue
            out.append(tool)
        return out

    # ── Prompt-token serializer ─────────────────────────────────────────────

    @staticmethod
    def serialize_for_prompt(tools: list[ToolSchema]) -> str:
        """
        Render a stable, deterministic block listing the resolved tools.

        Stable across calls for the same tools (sorted by qualified name) so
        the worker's system-prompt token count is reproducible. Returns an
        empty string for an empty tool list — no heading is emitted at all,
        keeping the byte cost truly zero when nothing is resolved.
        """
        if not tools:
            return ""
        ordered = sorted(tools, key=lambda t: (t.server_id, t.name))
        body = [
            {
                "tool":         f"{t.server_id}__{t.name}",
                "description":  t.description,
                "input_schema": t.input_schema,
            }
            for t in ordered
        ]
        # Sort_keys keeps property order stable inside each tool block.
        return PROMPT_HEADING + "\n```json\n" + json.dumps(
            body, indent=2, sort_keys=True,
        ) + "\n```"

    # ── Anthropic tools-array shape (deferred execution; surfaced for tests) ─

    def to_anthropic_tools(self, tools: list[ToolSchema]) -> list[dict]:
        return [t.to_anthropic_dict() for t in tools]
