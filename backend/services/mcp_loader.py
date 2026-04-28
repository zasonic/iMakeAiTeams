"""
services/mcp_loader.py — Phase 2: manifest discovery and folder ingest.

Each MCP server lives at::

    <mcp_servers_dir()>/<server_id>/mcp.json

Manifest schema (validated by ``parse_manifest``)::

    {
      "server_id":  "kebab-case-id",
      "name":       "Display Name",
      "version":    "1.0.0",
      "command":    "node",          # stored for future execution; unused now
      "args":       ["server.js"],
      "env_keys":   ["MY_TOKEN"],    # secret keys this server requires
      "tools": [
        { "name":         "search_files",
          "description":  "...",
          "input_schema": { "type": "object", "properties": {...} },
          "skill_tags":   ["researcher"],
          "scopes":       ["read"] }
      ]
    }

Malformed manifests are skipped with a warning — never raise to the caller.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

log = logging.getLogger("MyAIEnv.mcp_loader")

MANIFEST_FILENAME = "mcp.json"
_VALID_SERVER_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def validate_server_id(server_id: str) -> str:
    """Return the stripped server_id if it matches the manifest schema.

    Raises ``ValueError`` for any value that could escape the per-server
    namespace (path traversal in folder operations, keyring service-name
    injection, etc.). Use this at every API boundary that accepts a server_id
    from the renderer or HTTP routes.
    """
    if not isinstance(server_id, str):
        raise ValueError("server_id must be a string")
    cleaned = server_id.strip()
    if not _VALID_SERVER_ID.match(cleaned):
        raise ValueError(f"Invalid server_id {server_id!r}")
    return cleaned


@dataclass(frozen=True)
class ToolSchema:
    """A single MCP tool descriptor as resolved from a server manifest."""
    server_id:    str
    name:         str
    description:  str
    input_schema: dict
    skill_tags:   tuple[str, ...] = ()
    scopes:       tuple[str, ...] = ()

    def to_anthropic_dict(self) -> dict:
        """Render in the shape expected by Anthropic's ``tools`` parameter."""
        return {
            "name":         f"{self.server_id}__{self.name}",
            "description":  self.description,
            "input_schema": self.input_schema,
        }


@dataclass(frozen=True)
class MCPServer:
    """One discovered MCP server."""
    server_id:    str
    name:         str
    version:      str
    folder:       Path
    command:      str = ""
    args:         tuple[str, ...] = ()
    env_keys:     tuple[str, ...] = ()
    tools:        tuple[ToolSchema, ...] = field(default_factory=tuple)

    def tool_count(self) -> int:
        return len(self.tools)


class ManifestError(ValueError):
    """Raised by parse_manifest when a manifest is invalid."""


# ── Manifest parsing ─────────────────────────────────────────────────────────

def parse_manifest(folder: Path) -> MCPServer:
    """
    Parse and validate a server's mcp.json. Raises ManifestError on any issue.

    Callers that want non-raising semantics should use ``scan_servers()``.
    """
    manifest_path = folder / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise ManifestError(f"Missing {MANIFEST_FILENAME} in {folder}")
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"Cannot read {manifest_path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Invalid JSON in {manifest_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError(f"Manifest root must be a JSON object: {manifest_path}")

    server_id = str(data.get("server_id", "")).strip()
    name = str(data.get("name", "")).strip()
    if not server_id:
        raise ManifestError(f"Manifest missing 'server_id': {manifest_path}")
    if not _VALID_SERVER_ID.match(server_id):
        raise ManifestError(
            f"Invalid server_id {server_id!r} (lowercase alnum, _ and - only, "
            f"max 64 chars): {manifest_path}"
        )
    if not name:
        raise ManifestError(f"Manifest missing 'name': {manifest_path}")

    raw_tools = data.get("tools")
    if not isinstance(raw_tools, list) or not raw_tools:
        raise ManifestError(f"Manifest must declare a non-empty 'tools' list: {manifest_path}")

    tools: list[ToolSchema] = []
    for idx, t in enumerate(raw_tools):
        if not isinstance(t, dict):
            raise ManifestError(f"Tool #{idx} is not an object in {manifest_path}")
        t_name = str(t.get("name", "")).strip()
        t_desc = str(t.get("description", "")).strip()
        t_schema = t.get("input_schema")
        if not t_name:
            raise ManifestError(f"Tool #{idx} missing 'name' in {manifest_path}")
        if not isinstance(t_schema, dict):
            raise ManifestError(
                f"Tool {t_name!r} has non-object 'input_schema' in {manifest_path}"
            )
        tools.append(ToolSchema(
            server_id=server_id,
            name=t_name,
            description=t_desc,
            input_schema=t_schema,
            skill_tags=tuple(str(s) for s in (t.get("skill_tags") or [])),
            scopes=tuple(str(s) for s in (t.get("scopes") or [])),
        ))

    return MCPServer(
        server_id=server_id,
        name=name,
        version=str(data.get("version", "")).strip() or "0.0.0",
        folder=folder,
        command=str(data.get("command", "")).strip(),
        args=tuple(str(a) for a in (data.get("args") or [])),
        env_keys=tuple(str(k) for k in (data.get("env_keys") or [])),
        tools=tuple(tools),
    )


# ── Directory scanning ──────────────────────────────────────────────────────

def scan_servers(root: Path) -> list[MCPServer]:
    """
    Discover all valid MCP servers under ``root``. Malformed manifests are
    skipped with a logged warning (never raised). Returns empty list when the
    root does not exist.
    """
    if not root.exists() or not root.is_dir():
        return []

    discovered: list[MCPServer] = []
    seen_ids: set[str] = set()
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        try:
            server = parse_manifest(entry)
        except ManifestError as exc:
            log.warning("Skipping MCP server folder %s: %s", entry.name, exc)
            continue
        if server.server_id in seen_ids:
            log.warning(
                "Duplicate server_id %s in %s — keeping first occurrence",
                server.server_id, entry,
            )
            continue
        seen_ids.add(server.server_id)
        discovered.append(server)
    return discovered


# ── Folder ingest (copy a user-chosen folder into mcp_servers_dir) ──────────

@dataclass(frozen=True)
class IngestResult:
    server_id: str
    name:      str
    overwritten: bool


class IngestError(RuntimeError):
    """Raised by ingest_folder when the source folder cannot be installed."""


def ingest_folder(source: Path, root: Path, *, overwrite: bool = False) -> IngestResult:
    """
    Copy a user-chosen MCP server folder into ``root``/<server_id>/.

    Validates the source's mcp.json before any copy. If ``overwrite`` is False
    and a server with the same id already exists, raises IngestError so the
    UI can prompt the user to confirm.
    """
    source = Path(source).expanduser().resolve()
    if not source.exists() or not source.is_dir():
        raise IngestError(f"Source folder does not exist: {source}")

    try:
        server = parse_manifest(source)
    except ManifestError as exc:
        raise IngestError(f"Source folder is not a valid MCP server: {exc}") from exc

    target = Path(root) / server.server_id
    overwritten = False
    if target.exists():
        if not overwrite:
            raise IngestError(
                f"An MCP server with id {server.server_id!r} is already installed."
            )
        shutil.rmtree(target)
        overwritten = True

    shutil.copytree(source, target)
    # Touch the root so registries that poll mtime see the change immediately.
    try:
        Path(root).touch(exist_ok=True)
    except OSError:
        pass
    return IngestResult(
        server_id=server.server_id,
        name=server.name,
        overwritten=overwritten,
    )


def remove_server(server_id: str, root: Path) -> bool:
    """Delete an installed server folder. Returns True if anything was removed."""
    server_id = validate_server_id(server_id)
    target = Path(root) / server_id
    if not target.exists():
        return False
    shutil.rmtree(target)
    try:
        Path(root).touch(exist_ok=True)
    except OSError:
        pass
    return True


def collect_env_keys(servers: Iterable[MCPServer]) -> dict[str, list[str]]:
    """Helper for the UI: {server_id: [env_key, ...]}."""
    return {s.server_id: list(s.env_keys) for s in servers}
