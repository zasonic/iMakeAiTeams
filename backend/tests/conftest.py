"""
tests/conftest.py — Shared fixtures for the iMakeAiTeams backend tests.

Sets up an in-memory SQLite database and path hacks so that test files
can import backend modules without installing the package.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
# Add the backend directory to sys.path so that imports like "import db" work.
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))


# ── Disable OS keyring inside the test process ────────────────────────────────
# The legacy v5 tests pre-date the SECRET_KEYS / keyring routing in
# core/settings.py. They write plaintext API keys into settings.json and read
# them back from disk; with keyring enabled, those values get migrated to the
# OS keyring on load and cleared from JSON, breaking the assertions. The
# autouse fixture below force-disables both directions of the keyring helper
# so the legacy tests behave exactly as they did against the v5 layout.

@pytest.fixture(autouse=True)
def _disable_keyring(monkeypatch):
    from core import settings as _settings
    monkeypatch.setattr(_settings, "_keyring_get", lambda _key: None)
    monkeypatch.setattr(_settings, "_keyring_set", lambda _key, _value: False)
    monkeypatch.setattr(_settings, "_keyring_delete", lambda _key: None)


# ── In-memory DB fixture ──────────────────────────────────────────────────────

@pytest.fixture
def in_memory_db(tmp_path):
    """
    Initialise db with a fresh in-memory (tmp) SQLite DB.
    Tears down by resetting the module-level state.
    """
    import db
    db.init_db(tmp_path / "imakeaiteams.db")
    yield db
    # Teardown: close connection and reset globals
    if db._conn is not None:
        db._conn.close()
        db._conn = None
    db._db_path = None


# ── Mock Anthropic SDK ────────────────────────────────────────────────────────

@pytest.fixture
def mock_anthropic():
    """
    Patch anthropic.Anthropic so no real API calls are made.
    Returns the mock class; tests can configure return values as needed.
    """
    with patch("anthropic.Anthropic") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


# ── Minimal ClaudeClient fixture ─────────────────────────────────────────────

@pytest.fixture
def claude_client(mock_anthropic):
    """A ClaudeClient wired to the mock Anthropic instance."""
    from services.claude_client import ClaudeClient
    client = ClaudeClient.__new__(ClaudeClient)
    client._client = mock_anthropic
    client._model = "claude-sonnet-4-20250514"
    client._max_retries = 1
    # Replace the real worker methods with MagicMocks so tests can use
    # `.assert_called_once()` / `.assert_not_called()` on them. Tests that
    # need a particular return value override .return_value themselves.
    client.chat_multi_turn = MagicMock(return_value={
        "text": "claude reply", "input_tokens": 1, "output_tokens": 1,
    })
    _stream_usage = MagicMock(input_tokens=0, output_tokens=0)
    client.stream_multi_turn = MagicMock(return_value=("claude streamed", _stream_usage))
    return client


# ── Minimal LocalClient fixture ──────────────────────────────────────────────

@pytest.fixture
def local_client_unavailable():
    """A local client that always reports unavailable."""
    client = MagicMock()
    client.is_available.return_value = False
    return client


@pytest.fixture
def local_client_available():
    """A local client that reports available and returns canned responses."""
    client = MagicMock()
    client.is_available.return_value = True
    client.chat.return_value = '["test fact"]'
    client.chat_multi_turn.return_value = "local response"
    client.stream_multi_turn.return_value = ("local streamed response", None)
    client.chat_unified.return_value = {
        "text": "local response", "input_tokens": 0, "output_tokens": 0,
    }
    client.stream_unified.return_value = {
        "text": "local streamed response", "input_tokens": 0, "output_tokens": 0,
    }
    client.client_name.return_value = "local"
    return client


# ── Minimal Settings fixture ──────────────────────────────────────────────────

@pytest.fixture
def settings(tmp_path):
    """A real Settings instance backed by a temp file."""
    from core.settings import Settings
    return Settings(tmp_path / "settings.json")
