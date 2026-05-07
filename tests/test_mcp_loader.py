"""
tests/test_mcp_loader.py — Phase 2: manifest discovery + folder ingest.

Covers:
  - parse_manifest validation (server_id format, required fields, tool shape)
  - scan_servers skips malformed manifests with a warning, never raises
  - ingest_folder copies a valid source into mcp_servers_dir()
  - ingest_folder rejects existing server_id unless overwrite=True
  - hot-add: scan picks up a freshly-added server folder without restart
  - remove_server deletes a folder
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.mcp_loader import (
    IngestError,
    ManifestError,
    ingest_folder,
    parse_manifest,
    remove_server,
    scan_servers,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

VALID_MANIFEST = {
    "server_id": "demo-server",
    "name":      "Demo Server",
    "version":   "0.1.0",
    "command":   "node",
    "args":      ["server.js"],
    "env_keys":  ["DEMO_TOKEN"],
    "tools": [
        {"name":         "echo",
         "description":  "Echoes its input back.",
         "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
         "skill_tags":   ["researcher"],
         "scopes":       ["read"]},
    ],
}


def _write_server(folder: Path, manifest: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "mcp.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


# ── parse_manifest ──────────────────────────────────────────────────────────

class TestParseManifest:
    def test_valid_manifest(self, tmp_path):
        folder = _write_server(tmp_path / "demo-server", VALID_MANIFEST)
        server = parse_manifest(folder)
        assert server.server_id == "demo-server"
        assert server.name == "Demo Server"
        assert server.tool_count() == 1
        assert server.tools[0].name == "echo"
        assert server.env_keys == ("DEMO_TOKEN",)

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(ManifestError):
            parse_manifest(tmp_path / "nonexistent")

    def test_invalid_json_raises(self, tmp_path):
        folder = tmp_path / "bad-json"
        folder.mkdir()
        (folder / "mcp.json").write_text("{not valid", encoding="utf-8")
        with pytest.raises(ManifestError):
            parse_manifest(folder)

    def test_missing_server_id_raises(self, tmp_path):
        bad = dict(VALID_MANIFEST)
        bad.pop("server_id")
        folder = _write_server(tmp_path / "x", bad)
        with pytest.raises(ManifestError, match="server_id"):
            parse_manifest(folder)

    def test_invalid_server_id_format_raises(self, tmp_path):
        bad = dict(VALID_MANIFEST)
        bad["server_id"] = "Bad Id!"
        folder = _write_server(tmp_path / "x", bad)
        with pytest.raises(ManifestError, match="Invalid server_id"):
            parse_manifest(folder)

    def test_missing_tools_raises(self, tmp_path):
        bad = dict(VALID_MANIFEST)
        bad["tools"] = []
        folder = _write_server(tmp_path / "x", bad)
        with pytest.raises(ManifestError, match="tools"):
            parse_manifest(folder)

    def test_tool_missing_input_schema_raises(self, tmp_path):
        bad = dict(VALID_MANIFEST)
        bad["tools"] = [{"name": "x", "description": "y", "input_schema": "not-a-dict"}]
        folder = _write_server(tmp_path / "x", bad)
        with pytest.raises(ManifestError, match="input_schema"):
            parse_manifest(folder)


# ── scan_servers ────────────────────────────────────────────────────────────

class TestScanServers:
    def test_empty_root_returns_empty(self, tmp_path):
        assert scan_servers(tmp_path / "absent") == []
        assert scan_servers(tmp_path) == []

    def test_skips_malformed_without_raising(self, tmp_path):
        _write_server(tmp_path / "good", VALID_MANIFEST)
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "mcp.json").write_text("garbage", encoding="utf-8")
        # Also a folder with no manifest at all
        (tmp_path / "no-manifest").mkdir()
        results = scan_servers(tmp_path)
        assert len(results) == 1
        assert results[0].server_id == "demo-server"

    def test_duplicate_server_id_keeps_first(self, tmp_path):
        _write_server(tmp_path / "a", VALID_MANIFEST)
        _write_server(tmp_path / "b", VALID_MANIFEST)
        results = scan_servers(tmp_path)
        assert len(results) == 1


# ── ingest_folder + hot-add ─────────────────────────────────────────────────

class TestIngest:
    def test_ingest_copies_folder(self, tmp_path):
        source = _write_server(tmp_path / "src", VALID_MANIFEST)
        root = tmp_path / "installed"
        root.mkdir()
        result = ingest_folder(source, root)
        assert result.server_id == "demo-server"
        assert (root / "demo-server" / "mcp.json").exists()
        assert result.overwritten is False

    def test_ingest_rejects_existing_without_overwrite(self, tmp_path):
        source = _write_server(tmp_path / "src", VALID_MANIFEST)
        root = tmp_path / "installed"
        root.mkdir()
        ingest_folder(source, root)
        with pytest.raises(IngestError, match="already installed"):
            ingest_folder(source, root)

    def test_ingest_overwrites_when_requested(self, tmp_path):
        source = _write_server(tmp_path / "src", VALID_MANIFEST)
        root = tmp_path / "installed"
        root.mkdir()
        ingest_folder(source, root)
        result = ingest_folder(source, root, overwrite=True)
        assert result.overwritten is True

    def test_ingest_rejects_invalid_source(self, tmp_path):
        source = tmp_path / "src"
        source.mkdir()
        (source / "mcp.json").write_text("not json", encoding="utf-8")
        root = tmp_path / "installed"
        root.mkdir()
        with pytest.raises(IngestError):
            ingest_folder(source, root)

    def test_remove_server(self, tmp_path):
        source = _write_server(tmp_path / "src", VALID_MANIFEST)
        root = tmp_path / "installed"
        root.mkdir()
        ingest_folder(source, root)
        assert remove_server("demo-server", root) is True
        assert not (root / "demo-server").exists()
        assert remove_server("demo-server", root) is False


def test_hot_add_drop_in_picks_up_without_restart(tmp_path):
    """Drop a manifest into the scan root; the next scan must include it."""
    root = tmp_path / "installed"
    root.mkdir()
    assert scan_servers(root) == []

    # Simulate the user dropping a folder in (no app restart).
    _write_server(root / "demo-server", VALID_MANIFEST)
    after = scan_servers(root)
    assert len(after) == 1
    assert after[0].server_id == "demo-server"
