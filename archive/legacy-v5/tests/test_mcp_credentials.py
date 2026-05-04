"""
tests/test_mcp_credentials.py — Phase 2: per-MCP-server secret broker.

Covers:
  - set/get/delete round-trip via a fake keyring backend
  - list_set_secrets reports presence without exposing values
  - with_secrets context manager scopes secrets and clears on exit
  - **AST guard**: no module under app/services/ other than
    mcp_credentials.py reads the iMakeAiTeams.mcp.* keyring namespace
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from services import mcp_credentials


# ── In-memory keyring stub ──────────────────────────────────────────────────

class _FakeKeyring:
    def __init__(self):
        self._store: dict[tuple[str, str], str] = {}

    def set_password(self, service, key, value):
        self._store[(service, key)] = value

    def get_password(self, service, key):
        return self._store.get((service, key))

    def delete_password(self, service, key):
        self._store.pop((service, key), None)


@pytest.fixture
def fake_keyring(monkeypatch):
    fake = _FakeKeyring()
    monkeypatch.setitem(sys.modules, "keyring", fake)
    yield fake
    monkeypatch.delitem(sys.modules, "keyring", raising=False)


# ── Round-trip ──────────────────────────────────────────────────────────────


class TestCRUD:
    def test_set_get(self, fake_keyring):
        assert mcp_credentials.set_secret("demo", "API_KEY", "abc123") is True
        assert mcp_credentials.get_secret("demo", "API_KEY") == "abc123"

    def test_get_missing_returns_none(self, fake_keyring):
        assert mcp_credentials.get_secret("demo", "MISSING") is None

    def test_delete(self, fake_keyring):
        mcp_credentials.set_secret("demo", "API_KEY", "abc123")
        mcp_credentials.delete_secret("demo", "API_KEY")
        assert mcp_credentials.get_secret("demo", "API_KEY") is None

    def test_isolation_between_servers(self, fake_keyring):
        mcp_credentials.set_secret("a", "TOK", "one")
        mcp_credentials.set_secret("b", "TOK", "two")
        assert mcp_credentials.get_secret("a", "TOK") == "one"
        assert mcp_credentials.get_secret("b", "TOK") == "two"

    def test_empty_server_id_rejected(self, fake_keyring):
        with pytest.raises(ValueError):
            mcp_credentials.set_secret("", "X", "y")

    def test_keyring_failure_is_silent(self, monkeypatch):
        # No 'keyring' module — set returns False, get returns None.
        monkeypatch.setitem(sys.modules, "keyring", None)
        assert mcp_credentials.set_secret("demo", "X", "y") is False
        assert mcp_credentials.get_secret("demo", "X") is None


# ── Presence-only listing ───────────────────────────────────────────────────


class TestListSetSecrets:
    def test_reports_presence_only(self, fake_keyring):
        mcp_credentials.set_secret("demo", "A", "value-a")
        out = mcp_credentials.list_set_secrets("demo", ["A", "B"])
        assert out == {"A": True, "B": False}
        # Result must not contain the secret values themselves
        assert "value-a" not in str(out)


# ── Context manager scoping ─────────────────────────────────────────────────


class TestWithSecrets:
    def test_yields_present_secrets(self, fake_keyring):
        mcp_credentials.set_secret("demo", "X", "xv")
        with mcp_credentials.with_secrets("demo", ["X", "MISSING"]) as env:
            assert env == {"X": "xv"}

    def test_clears_on_exit(self, fake_keyring):
        mcp_credentials.set_secret("demo", "X", "xv")
        with mcp_credentials.with_secrets("demo", ["X"]) as env:
            captured = env  # alias
        assert captured == {}  # cleared


# ── AST guard: only mcp_credentials.py touches the namespace ────────────────


def test_no_other_service_reads_mcp_keyring_namespace():
    """
    Static check: in app/services/, only mcp_credentials.py may construct a
    service name beginning with 'iMakeAiTeams.mcp.' (the prefix used to scope
    MCP secrets). Other code must go through this module.
    """
    services_dir = Path(__file__).parent.parent / "app" / "services"
    needle = "iMakeAiTeams.mcp."
    allowed = {"mcp_credentials.py"}
    offenders: list[str] = []
    for py in services_dir.glob("*.py"):
        if py.name in allowed:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if needle in node.value:
                    offenders.append(f"{py.name}:{node.lineno}")
    assert not offenders, (
        "Other services reference the MCP keyring namespace directly:\n  "
        + "\n  ".join(offenders)
    )
