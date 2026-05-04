"""
tests/conftest.py — Shared fixtures for MyAI Agent Hub test suite.

Sets up an in-memory SQLite database and path hacks so that test files
can import app modules without installing the package.
"""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
# Add the app directory to sys.path so that imports like "import db" work.
APP_DIR = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(APP_DIR))


# ── In-memory DB fixture ──────────────────────────────────────────────────────

@pytest.fixture
def in_memory_db(tmp_path):
    """
    Initialise db with a fresh in-memory (tmp) SQLite DB.
    Tears down by resetting the module-level state.
    """
    import db
    db.init_db(tmp_path / "myai.db")
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
    client.stream_multi_turn.return_value = "local streamed response"
    return client


# ── Minimal Settings fixture ──────────────────────────────────────────────────

@pytest.fixture
def settings(tmp_path):
    """A real Settings instance backed by a temp file."""
    from core.settings import Settings
    return Settings(tmp_path / "settings.json")
