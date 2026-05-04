"""
tests/test_mcp_registry.py — Phase 2: in-memory tool catalog.

Covers the four success criteria from the Phase 2 plan:
  1. resolve_for_task: skill-tag intersection, scope subset
  2. **token-budget invariance**: serialize_for_prompt size depends only on the
     resolved subset, not on total catalog size
  3. enable/disable toggle filters tools out
  4. mtime-based hot-reload picks up new servers without process restart
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from services.mcp_registry import MCPRegistry


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_manifest(server_id: str, tool_count: int) -> dict:
    return {
        "server_id": server_id,
        "name":      f"Server {server_id}",
        "version":   "1.0.0",
        "tools": [
            {"name":         f"tool_{i}",
             "description":  f"Tool {i} from {server_id}",
             "input_schema": {"type": "object",
                              "properties": {"q": {"type": "string"}}},
             "skill_tags":   ["researcher"],
             "scopes":       ["read"]}
            for i in range(tool_count)
        ],
    }


def _seed(root: Path, server_id: str, tool_count: int = 1) -> None:
    folder = root / server_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "mcp.json").write_text(
        json.dumps(_make_manifest(server_id, tool_count)), encoding="utf-8",
    )


@pytest.fixture
def registry(tmp_path, settings):
    root = tmp_path / "mcp"
    root.mkdir()
    return MCPRegistry(root, settings), root


# ── Discovery + listing ─────────────────────────────────────────────────────

class TestListing:
    def test_empty_root(self, registry):
        reg, _ = registry
        assert reg.list_servers() == []
        assert reg.all_tools() == []

    def test_lists_seeded_servers(self, registry):
        reg, root = registry
        _seed(root, "alpha", 2)
        _seed(root, "beta", 3)
        servers = reg.list_servers()
        assert {s.server_id for s in servers} == {"alpha", "beta"}
        assert len(reg.all_tools()) == 5


# ── resolve_for_task ────────────────────────────────────────────────────────

class TestResolve:
    def test_returns_only_matching_skill_tags(self, registry):
        reg, root = registry
        _seed(root, "alpha", 1)
        # Add a server whose tools have a different skill tag
        folder = root / "writer-server"
        folder.mkdir()
        m = _make_manifest("writer-server", 1)
        m["tools"][0]["skill_tags"] = ["writer"]
        (folder / "mcp.json").write_text(json.dumps(m), encoding="utf-8")

        out = reg.resolve_for_task(["researcher"])
        assert len(out) == 1
        assert out[0].server_id == "alpha"

    def test_no_skills_returns_empty(self, registry):
        reg, root = registry
        _seed(root, "alpha", 1)
        assert reg.resolve_for_task([]) == []

    def test_scopes_must_be_subset(self, registry):
        reg, root = registry
        # Tool declares scopes=["read"]; a write request must be rejected.
        _seed(root, "alpha", 1)
        assert reg.resolve_for_task(["researcher"], required_scopes=["write"]) == []
        assert len(reg.resolve_for_task(["researcher"], required_scopes=["read"])) == 1

    def test_excludes_tools_with_no_skill_tags(self, tmp_path, settings):
        root = tmp_path / "mcp"
        root.mkdir()
        folder = root / "untagged"
        folder.mkdir()
        m = _make_manifest("untagged", 1)
        m["tools"][0]["skill_tags"] = []
        (folder / "mcp.json").write_text(json.dumps(m), encoding="utf-8")
        reg = MCPRegistry(root, settings)
        assert reg.resolve_for_task(["researcher"]) == []


# ── Enable/disable ──────────────────────────────────────────────────────────

class TestEnableDisable:
    def test_disabled_server_excluded_from_all_tools(self, registry):
        reg, root = registry
        _seed(root, "alpha", 1)
        _seed(root, "beta", 1)
        reg.set_enabled("beta", False)
        ids = {t.server_id for t in reg.all_tools()}
        assert ids == {"alpha"}

    def test_disabled_server_excluded_from_resolve(self, registry):
        reg, root = registry
        _seed(root, "alpha", 1)
        reg.set_enabled("alpha", False)
        assert reg.resolve_for_task(["researcher"]) == []

    def test_re_enable_includes_again(self, registry):
        reg, root = registry
        _seed(root, "alpha", 1)
        reg.set_enabled("alpha", False)
        assert reg.resolve_for_task(["researcher"]) == []
        reg.set_enabled("alpha", True)
        assert len(reg.resolve_for_task(["researcher"])) == 1


# ── Token-budget invariance (Success criterion 1) ───────────────────────────


def _resolved_size_for_catalog(tmp_path: Path, settings, total_servers: int) -> int:
    """
    Build a catalog of N servers (each with one matching tool plus one
    non-matching tool) and return the serialized prompt-tools byte size.

    The resolved set should be exactly N matching tools when we ask for the
    matching skill, regardless of how many non-matching tools exist. So this
    helper proves invariance against irrelevant catalog growth: callers can
    fix N (e.g. always 1 matching tool) and vary the non-matching pool.
    """
    root = tmp_path / f"cat-{total_servers}"
    root.mkdir()
    # Always one server with one *matching* tool
    folder = root / "alpha"
    folder.mkdir()
    m = _make_manifest("alpha", 1)  # single tool, skill_tags=["researcher"]
    (folder / "mcp.json").write_text(json.dumps(m), encoding="utf-8")
    # Plus N-1 servers with non-matching tools
    for i in range(1, total_servers):
        f = root / f"noise-{i:03d}"
        f.mkdir()
        nm = _make_manifest(f"noise-{i:03d}", 5)
        for t in nm["tools"]:
            t["skill_tags"] = ["unrelated"]
        (f / "mcp.json").write_text(json.dumps(nm), encoding="utf-8")
    reg = MCPRegistry(root, settings)
    resolved = reg.resolve_for_task(["researcher"], required_scopes=["read"])
    return len(MCPRegistry.serialize_for_prompt(resolved).encode("utf-8"))


def test_prompt_token_invariance(tmp_path, settings):
    """
    Worker prompt-tool block size must NOT change as the total catalog grows,
    so long as the resolved-tool subset is identical. Tests catalogs of
    1, 10, 100 total servers — same single matching tool — same size.
    """
    s1   = _resolved_size_for_catalog(tmp_path, settings, 1)
    s10  = _resolved_size_for_catalog(tmp_path, settings, 10)
    s100 = _resolved_size_for_catalog(tmp_path, settings, 100)
    assert s1 == s10 == s100, (
        f"Prompt-tool block size grew with catalog: {s1} → {s10} → {s100}. "
        "The resolved subset is identical across catalogs, so the serialized "
        "prompt MUST be identical."
    )


def test_empty_resolution_emits_zero_bytes(tmp_path, settings):
    """No matching tools → no prompt block at all (true zero cost)."""
    root = tmp_path / "mcp"
    root.mkdir()
    _seed(root, "alpha", 1)  # tool tagged 'researcher'
    reg = MCPRegistry(root, settings)
    out = reg.serialize_for_prompt(reg.resolve_for_task(["nonexistent"]))
    assert out == ""


# ── Hot-reload (Success criterion 2) ────────────────────────────────────────


def test_hot_add_within_one_resolve_call(tmp_path, settings):
    """
    Drop a server folder into the scan root and the very next resolve must
    pick it up with no explicit refresh and no app restart.
    """
    root = tmp_path / "mcp"
    root.mkdir()
    reg = MCPRegistry(root, settings)
    assert reg.list_servers() == []

    # Sleep briefly so the directory mtime advances detectably on filesystems
    # with second-granularity (e.g. older FAT/HFS).
    time.sleep(0.01)
    _seed(root, "newcomer", 1)
    # Touch the directory mtime so mtime polling fires even on coarse FS clocks.
    Path(root).touch()

    servers = reg.list_servers()
    assert len(servers) == 1 and servers[0].server_id == "newcomer"


# ── Anthropic shape (sanity) ────────────────────────────────────────────────


def test_to_anthropic_tools_namespaces_with_server(registry):
    reg, root = registry
    _seed(root, "alpha", 1)
    tools = reg.resolve_for_task(["researcher"])
    shaped = reg.to_anthropic_tools(tools)
    assert shaped[0]["name"] == "alpha__tool_0"
    assert "input_schema" in shaped[0]
