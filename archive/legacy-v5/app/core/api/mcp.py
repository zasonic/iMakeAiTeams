"""
core/api/mcp.py — JS-API surface for the MCP tool registry (Phase 2).

Exposes installed-server inspection, folder-picker ingest, removal, and
per-server enable/disable to the PyWebView frontend. Tool execution is
deferred to a later phase; this module never invokes a server, only manages
the catalog.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core import paths
from services import mcp_credentials
from services.mcp_loader import IngestError, ingest_folder, remove_server

from ._base import BaseAPI

log = logging.getLogger("MyAIEnv.api.mcp")


class MCPAPI(BaseAPI):

    # ── Catalog views ────────────────────────────────────────────────────────

    def list_mcp_servers(self) -> dict:
        """Return installed servers with their tool counts and enable state."""
        registry = self._mcp_registry
        servers = registry.list_servers()
        out = []
        for s in servers:
            secrets = mcp_credentials.list_set_secrets(s.server_id, list(s.env_keys))
            out.append({
                "server_id":   s.server_id,
                "name":        s.name,
                "version":     s.version,
                "tool_count":  s.tool_count(),
                "enabled":     registry.is_enabled(s.server_id),
                "env_keys":    list(s.env_keys),
                "env_set":     secrets,
                "tools": [
                    {
                        "name":        t.name,
                        "description": t.description,
                        "skill_tags":  list(t.skill_tags),
                        "scopes":      list(t.scopes),
                    }
                    for t in s.tools
                ],
            })
        return {"servers": out, "root": str(paths.mcp_servers_dir())}

    # ── Folder-picker ingest ─────────────────────────────────────────────────

    def pick_mcp_server_folder(self, *, overwrite: bool = False) -> dict:
        """Open a native folder picker, validate, and copy under mcp_servers_dir().

        Returns ``{"ok": True, "server_id": ..., "name": ..., "overwritten": bool}``
        on success, ``{"ok": False, "cancelled": True}`` if the user dismissed
        the dialog, or ``{"ok": False, "error": str, "needs_overwrite_confirm":
        bool}`` on a validation/copy failure.
        """
        try:
            import webview as _wv
        except Exception as exc:
            return {"ok": False, "error": f"PyWebView unavailable: {exc}"}
        if self._window is None:
            return {"ok": False, "error": "Window is not initialized."}
        result = self._window.create_file_dialog(
            _wv.FOLDER_DIALOG, allow_multiple=False,
        )
        if not result:
            return {"ok": False, "cancelled": True}
        chosen = Path(result[0])
        try:
            ingested = ingest_folder(
                chosen, paths.mcp_servers_dir(), overwrite=overwrite,
            )
        except IngestError as exc:
            return {
                "ok": False,
                "error": str(exc),
                "needs_overwrite_confirm": "already installed" in str(exc),
            }
        # Force a registry refresh so the next list_mcp_servers reflects it.
        self._mcp_registry.refresh()
        log.info(
            "Installed MCP server %s (overwrote=%s)",
            ingested.server_id, ingested.overwritten,
        )
        return {
            "ok":          True,
            "server_id":   ingested.server_id,
            "name":        ingested.name,
            "overwritten": ingested.overwritten,
        }

    # ── Removal / enable / disable ───────────────────────────────────────────

    def remove_mcp_server(self, server_id: str) -> dict:
        removed = remove_server(server_id, paths.mcp_servers_dir())
        if not removed:
            return {"ok": False, "error": "Server not found."}
        self._mcp_registry.refresh()
        return {"ok": True, "server_id": server_id}

    def set_mcp_server_enabled(self, server_id: str, enabled: bool) -> dict:
        self._mcp_registry.set_enabled(server_id, bool(enabled))
        return {"ok": True, "server_id": server_id, "enabled": bool(enabled)}

    # ── Per-server credentials ───────────────────────────────────────────────

    def set_mcp_secret(self, server_id: str, key: str, value: str) -> dict:
        if not key or not isinstance(key, str):
            return {"ok": False, "error": "key is required"}
        if not isinstance(value, str):
            return {"ok": False, "error": "value must be a string"}
        ok = mcp_credentials.set_secret(server_id, key, value)
        return {"ok": ok}

    def clear_mcp_secret(self, server_id: str, key: str) -> dict:
        mcp_credentials.delete_secret(server_id, key)
        return {"ok": True}

    # ── Manual refresh (rarely needed; mtime poll handles most cases) ────────

    def refresh_mcp_registry(self) -> dict:
        self._mcp_registry.refresh()
        return {"ok": True, "count": len(self._mcp_registry.list_servers())}
