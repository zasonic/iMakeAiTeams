"""tests/test_conversation_export.py — HTTP-level tests for the conversation
export routes added in PR 7.

Three formats are exposed off the existing /api/chat prefix:

    GET /api/chat/conversations/{id}/export.md       (text/markdown)
    GET /api/chat/conversations/{id}/export.json     (application/json)
    GET /api/chat/conversations/{id}/export.pdf-html (text/html)

These tests mount only the chat router on a minimal FastAPI app — the
sidecar's full container isn't needed because the export handlers read
straight from the SQLite layer (`db.fetchone` / `db.fetchall`).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routes import chat as chat_routes
from server import BearerAuthMiddleware


TOKEN = "test-token-export"


def _auth() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def app(in_memory_db):
    a = FastAPI()
    a.add_middleware(BearerAuthMiddleware, expected_token=TOKEN)
    a.include_router(chat_routes.router, prefix="/api/chat")
    return a


def _seed_conversation(in_memory_db, *, cid: str = "conv-1",
                       title: str = "Pancake recipes") -> None:
    """Seed a conversation with two user/assistant turns."""
    now = datetime.now(timezone.utc).isoformat()
    in_memory_db.execute(
        "INSERT INTO conversations (id, title, agent_id, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (cid, title, "", now, now),
    )
    msgs = [
        ("m1", "user",      "How do I make fluffy pancakes?"),
        ("m2", "assistant", "Whisk dry, then fold in wet — see ```recipe()```."),
        ("m3", "user",      "What about gluten-free?"),
        ("m4", "assistant", "Sub buckwheat flour 1:1 and add a pinch of xanthan."),
    ]
    for mid, role, content in msgs:
        in_memory_db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, "
            "model_used, route_reason, tokens_in, tokens_out, cost_usd, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (mid, cid, role, content, "claude-sonnet", "ok",
             10, 20, 0.0001, now),
        )
    in_memory_db.commit()


# ── Markdown ───────────────────────────────────────────────────────────────


class TestMarkdownExport:
    def test_returns_200_and_text_markdown(self, app, in_memory_db):
        _seed_conversation(in_memory_db)
        client = TestClient(app)
        resp = client.get(
            "/api/chat/conversations/conv-1/export.md",
            headers=_auth(),
        )
        assert resp.status_code == 200
        ctype = resp.headers["content-type"]
        assert ctype.startswith("text/markdown"), ctype

    def test_contains_exact_role_headers_and_title(self, app, in_memory_db):
        _seed_conversation(in_memory_db)
        client = TestClient(app)
        resp = client.get(
            "/api/chat/conversations/conv-1/export.md",
            headers=_auth(),
        )
        body = resp.text
        # Title line
        assert body.startswith("# Pancake recipes")
        # Exact role headers — these are the contract the renderer relies on
        assert "## You" in body
        assert "## Assistant" in body
        # Message bodies are preserved verbatim
        assert "How do I make fluffy pancakes?" in body
        assert "Sub buckwheat flour 1:1" in body
        # An "Exported …" stamp appears in the header section
        assert "Exported" in body


# ── JSON ───────────────────────────────────────────────────────────────────


class TestJsonExport:
    def test_returns_200_and_application_json(self, app, in_memory_db):
        _seed_conversation(in_memory_db)
        client = TestClient(app)
        resp = client.get(
            "/api/chat/conversations/conv-1/export.json",
            headers=_auth(),
        )
        assert resp.status_code == 200
        ctype = resp.headers["content-type"]
        assert ctype.startswith("application/json"), ctype

    def test_parses_as_a_list_of_message_records(self, app, in_memory_db):
        _seed_conversation(in_memory_db)
        client = TestClient(app)
        resp = client.get(
            "/api/chat/conversations/conv-1/export.json",
            headers=_auth(),
        )
        parsed = json.loads(resp.text)
        assert isinstance(parsed, list)
        assert len(parsed) == 4
        # Spot-check the raw record shape — same keys the rest of the app
        # already knows how to consume.
        first = parsed[0]
        assert first["role"] == "user"
        assert first["content"] == "How do I make fluffy pancakes?"
        assert "created_at" in first
        assert first["model_used"] == "claude-sonnet"


# ── PDF-html ───────────────────────────────────────────────────────────────


class TestPdfHtmlExport:
    def test_returns_200_and_text_html(self, app, in_memory_db):
        _seed_conversation(in_memory_db)
        client = TestClient(app)
        resp = client.get(
            "/api/chat/conversations/conv-1/export.pdf-html",
            headers=_auth(),
        )
        assert resp.status_code == 200
        ctype = resp.headers["content-type"]
        assert ctype.startswith("text/html"), ctype

    def test_contains_html_skeleton_and_message_content(self, app, in_memory_db):
        _seed_conversation(in_memory_db)
        client = TestClient(app)
        resp = client.get(
            "/api/chat/conversations/conv-1/export.pdf-html",
            headers=_auth(),
        )
        body = resp.text
        # The skeleton printToPDF expects
        assert "<html" in body
        assert "<head" in body
        assert 'charset="utf-8"' in body
        assert "<style>" in body
        assert "<body" in body
        # Message content escaped into the HTML
        assert "How do I make fluffy pancakes?" in body
        assert "Sub buckwheat flour 1:1" in body
        # Title rendered as the heading
        assert "Pancake recipes" in body


# ── Errors ─────────────────────────────────────────────────────────────────


class TestNotFound:
    @pytest.mark.parametrize("ext", ["md", "json", "pdf-html"])
    def test_unknown_conversation_returns_404(self, app, in_memory_db, ext):
        client = TestClient(app)
        resp = client.get(
            f"/api/chat/conversations/does-not-exist/export.{ext}",
            headers=_auth(),
        )
        assert resp.status_code == 404


class TestAuth:
    @pytest.mark.parametrize("ext", ["md", "json", "pdf-html"])
    def test_rejects_without_bearer_auth(self, app, in_memory_db, ext):
        _seed_conversation(in_memory_db)
        client = TestClient(app)
        resp = client.get(f"/api/chat/conversations/conv-1/export.{ext}")
        assert resp.status_code == 401
