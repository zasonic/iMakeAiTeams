"""tests/test_conversation_search.py — HTTP-level tests for the FTS5
cross-conversation search added in PR 13.

The route lives at GET /api/conversations/search and reads straight from
SQLite (db.fetchall) with the messages_fts virtual table created in
migration phase11.message_fts. These tests mount only the conversations
router on a minimal FastAPI app — the sidecar's container isn't needed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routes import conversations as conversations_routes
from server import BearerAuthMiddleware


TOKEN = "test-token-search"


def _auth() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def app(in_memory_db):
    a = FastAPI()
    a.add_middleware(BearerAuthMiddleware, expected_token=TOKEN)
    a.include_router(conversations_routes.router, prefix="/api/conversations")
    return a


def _seed(in_memory_db, *, cid: str, title: str, msgs: list[tuple], offset_days: int = 0) -> None:
    """Insert a conversation with messages dated ``offset_days`` ago."""
    base = datetime.now(timezone.utc) - timedelta(days=offset_days)
    in_memory_db.execute(
        "INSERT OR IGNORE INTO conversations (id, title, agent_id, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (cid, title, "", base.isoformat(), base.isoformat()),
    )
    for i, (mid, role, content) in enumerate(msgs):
        # Stagger created_at by milliseconds so ORDER BY is deterministic
        # for any tests that need it.
        ts = (base + timedelta(milliseconds=i)).isoformat()
        in_memory_db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, "
            "model_used, route_reason, tokens_in, tokens_out, cost_usd, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (mid, cid, role, content, "claude-sonnet", "", 0, 0, 0.0, ts),
        )
    in_memory_db.commit()


# ── Migration sanity ─────────────────────────────────────────────────────────


class TestMigration:
    def test_messages_fts_table_exists_and_is_backfilled(self, in_memory_db):
        _seed(
            in_memory_db,
            cid="c1",
            title="Pancakes",
            msgs=[
                ("m1", "user", "How do I make fluffy pancakes?"),
                ("m2", "assistant", "Whisk dry, then fold in wet."),
            ],
        )
        # The trigger inserts each row as it lands. Counting messages_fts
        # should match the messages count after seeding.
        n_msgs = in_memory_db.fetchone("SELECT COUNT(*) AS n FROM messages")["n"]
        n_fts = in_memory_db.fetchone("SELECT COUNT(*) AS n FROM messages_fts")["n"]
        assert n_msgs == 2
        assert n_fts == n_msgs


# ── Search behavior ──────────────────────────────────────────────────────────


class TestSearch:
    def test_returns_match_with_snippet_and_mark_tags(self, app, in_memory_db):
        _seed(
            in_memory_db,
            cid="c1",
            title="Pancakes",
            msgs=[
                ("m1", "user", "How do I make fluffy pancakes for breakfast?"),
                ("m2", "assistant", "Whisk dry, then fold in wet — see ```recipe()```."),
            ],
        )
        client = TestClient(app)
        resp = client.get(
            "/api/conversations/search",
            params={"q": "pancakes"},
            headers=_auth(),
        )
        assert resp.status_code == 200
        rows = resp.json()
        assert isinstance(rows, list)
        assert len(rows) == 1
        first = rows[0]
        assert first["message_id"] == "m1"
        assert first["conversation_id"] == "c1"
        assert first["conversation_title"] == "Pancakes"
        assert first["role"] == "user"
        assert first["snippet"]
        assert "<mark>" in first["snippet"]
        assert "</mark>" in first["snippet"]
        assert "rank" in first
        assert "created_at" in first

    def test_results_ordered_by_bm25(self, app, in_memory_db):
        # Two messages, one with "pancake" once and one with "pancake"
        # repeated three times. BM25 should rank the denser match first.
        _seed(
            in_memory_db,
            cid="c1",
            title="Mixed",
            msgs=[
                ("m1", "user", "I like pancake recipes a lot."),
                ("m2", "assistant", "Pancake pancake pancake breakfast joy."),
            ],
        )
        client = TestClient(app)
        resp = client.get(
            "/api/conversations/search",
            params={"q": "pancake"},
            headers=_auth(),
        )
        assert resp.status_code == 200
        rows = resp.json()
        assert len(rows) == 2
        # bm25 returns smaller (more negative) values for better matches
        # and the route ORDERs by bm25 ASC, so the denser match comes
        # first.
        assert rows[0]["message_id"] == "m2"
        assert rows[1]["message_id"] == "m1"
        assert rows[0]["rank"] <= rows[1]["rank"]

    def test_days_filter_excludes_older_messages(self, app, in_memory_db):
        _seed(
            in_memory_db,
            cid="recent",
            title="Recent",
            msgs=[("mr", "user", "Modern pancake recipe.")],
            offset_days=1,
        )
        _seed(
            in_memory_db,
            cid="old",
            title="Ancient",
            msgs=[("mo", "user", "Vintage pancake recipe.")],
            offset_days=400,
        )
        client = TestClient(app)
        # Without filter both should land.
        all_resp = client.get(
            "/api/conversations/search",
            params={"q": "pancake"},
            headers=_auth(),
        )
        assert all_resp.status_code == 200
        assert len(all_resp.json()) == 2

        # days=30 should exclude the 400-day-old message.
        scoped_resp = client.get(
            "/api/conversations/search",
            params={"q": "pancake", "days": 30},
            headers=_auth(),
        )
        assert scoped_resp.status_code == 200
        scoped = scoped_resp.json()
        assert len(scoped) == 1
        assert scoped[0]["message_id"] == "mr"

    def test_limit_caps_results(self, app, in_memory_db):
        msgs = [(f"m{i}", "user", f"pancake number {i}") for i in range(5)]
        _seed(in_memory_db, cid="c1", title="Many", msgs=msgs)
        client = TestClient(app)
        resp = client.get(
            "/api/conversations/search",
            params={"q": "pancake", "limit": 2},
            headers=_auth(),
        )
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_default_limit_is_50(self, app, in_memory_db):
        msgs = [(f"m{i}", "user", f"pancake number {i}") for i in range(60)]
        _seed(in_memory_db, cid="c1", title="Lots", msgs=msgs)
        client = TestClient(app)
        resp = client.get(
            "/api/conversations/search",
            params={"q": "pancake"},
            headers=_auth(),
        )
        assert resp.status_code == 200
        assert len(resp.json()) == 50

    def test_limit_above_max_is_rejected(self, app, in_memory_db):
        client = TestClient(app)
        resp = client.get(
            "/api/conversations/search",
            params={"q": "pancake", "limit": 1000},
            headers=_auth(),
        )
        # FastAPI's Query(le=200) returns 422 for out-of-range params.
        assert resp.status_code == 422


# ── Errors ───────────────────────────────────────────────────────────────────


class TestQueryValidation:
    def test_empty_query_returns_400(self, app, in_memory_db):
        client = TestClient(app)
        resp = client.get(
            "/api/conversations/search",
            params={"q": "   "},
            headers=_auth(),
        )
        assert resp.status_code == 400
        assert "Invalid search query" in resp.text

    def test_invalid_fts5_syntax_returns_400(self, app, in_memory_db):
        # Seed at least one message so the FTS5 index isn't empty — empty
        # indexes can short-circuit before hitting the parser.
        _seed(
            in_memory_db,
            cid="c1",
            title="Stub",
            msgs=[("m1", "user", "anything")],
        )
        client = TestClient(app)
        # Unbalanced double quote — FTS5 parser raises OperationalError
        # which the route translates to 400.
        resp = client.get(
            "/api/conversations/search",
            params={"q": '"unbalanced'},
            headers=_auth(),
        )
        assert resp.status_code == 400
        assert "Invalid search query" in resp.text


# ── Triggers ─────────────────────────────────────────────────────────────────


class TestTriggers:
    def test_inserting_a_message_makes_it_findable(self, app, in_memory_db):
        # Start with no messages, then add one and confirm search finds it.
        # This exercises the messages_fts_insert trigger end-to-end.
        in_memory_db.execute(
            "INSERT INTO conversations (id, title, agent_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("c1", "TriggerTest", "", "2026-05-08T00:00:00+00:00", "2026-05-08T00:00:00+00:00"),
        )
        in_memory_db.commit()

        client = TestClient(app)
        before = client.get(
            "/api/conversations/search",
            params={"q": "zebracorn"},
            headers=_auth(),
        )
        assert before.status_code == 200
        assert before.json() == []

        in_memory_db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, "
            "model_used, route_reason, tokens_in, tokens_out, cost_usd, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("m-new", "c1", "user", "An elegant zebracorn appeared.",
             "claude-sonnet", "", 0, 0, 0.0, "2026-05-08T00:00:01+00:00"),
        )
        in_memory_db.commit()

        after = client.get(
            "/api/conversations/search",
            params={"q": "zebracorn"},
            headers=_auth(),
        )
        assert after.status_code == 200
        rows = after.json()
        assert len(rows) == 1
        assert rows[0]["message_id"] == "m-new"

    def test_deleting_a_message_removes_it_from_search(self, app, in_memory_db):
        _seed(
            in_memory_db,
            cid="c1",
            title="DeleteTest",
            msgs=[("m1", "user", "uniqueterm pancake apple")],
        )
        client = TestClient(app)
        before = client.get(
            "/api/conversations/search",
            params={"q": "uniqueterm"},
            headers=_auth(),
        )
        assert before.status_code == 200
        assert len(before.json()) == 1

        in_memory_db.execute("DELETE FROM messages WHERE id = ?", ("m1",))
        in_memory_db.commit()

        after = client.get(
            "/api/conversations/search",
            params={"q": "uniqueterm"},
            headers=_auth(),
        )
        assert after.status_code == 200
        assert after.json() == []


class TestAuth:
    def test_rejects_without_bearer_auth(self, app, in_memory_db):
        client = TestClient(app)
        resp = client.get("/api/conversations/search", params={"q": "anything"})
        assert resp.status_code == 401
