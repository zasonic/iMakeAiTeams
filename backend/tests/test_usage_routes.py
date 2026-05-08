"""tests/test_usage_routes.py — HTTP-level tests for /api/usage routes.

Exercises the read-only token-usage aggregation endpoint via FastAPI's
TestClient. Builds a minimal app over the in_memory_db fixture so the
queries hit a real (temp) SQLite file and exercise the same code path
the renderer talks to.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routes import usage as usage_routes
from server import BearerAuthMiddleware


TOKEN = "test-token-usage"


def _auth() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


@pytest.fixture
def app(in_memory_db):
    a = FastAPI()
    a.add_middleware(BearerAuthMiddleware, expected_token=TOKEN)
    a.include_router(usage_routes.router, prefix="/api/usage")
    return a


def _seed_usage(
    db,
    *,
    row_id: str,
    conversation_id: str | None,
    model: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
    created_at: str,
) -> None:
    db.execute(
        "INSERT INTO token_usage (id, conversation_id, model, tokens_in, "
        "tokens_out, cost_usd, routed_reason, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (row_id, conversation_id, model, tokens_in, tokens_out, cost_usd,
         "test", created_at),
    )
    db.commit()


def _seed_conversation(db, conv_id: str, agent_id: str | None) -> None:
    db.execute(
        "INSERT INTO conversations (id, title, agent_id, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (conv_id, "test", agent_id, _iso(datetime.now(timezone.utc)),
         _iso(datetime.now(timezone.utc))),
    )
    db.commit()


# ── Empty database ─────────────────────────────────────────────────────────

class TestEmptyDatabase:
    def test_returns_zeros_and_empty_rows(self, app):
        client = TestClient(app)
        resp = client.get("/api/usage/summary", headers=_auth())
        assert resp.status_code == 200
        body = resp.json()

        assert body["window_days"] == 30
        assert body["group_by"]    == "day"
        assert body["total"] == {
            "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "turns": 0,
        }
        assert body["rows"]     == []
        assert body["by_model"] == []
        assert body["by_agent"] == []


# ── Seeded database ────────────────────────────────────────────────────────

class TestSeededAggregation:
    def test_total_matches_sum_of_seeded_costs(self, app, in_memory_db):
        db = in_memory_db
        now = datetime.now(timezone.utc)
        _seed_conversation(db, "c1", "agent-A")
        _seed_conversation(db, "c2", "agent-B")
        _seed_usage(db, row_id="u1", conversation_id="c1", model="claude",
                    tokens_in=100, tokens_out=50, cost_usd=0.10,
                    created_at=_iso(now - timedelta(days=1)))
        _seed_usage(db, row_id="u2", conversation_id="c2", model="qwen3",
                    tokens_in=200, tokens_out=80, cost_usd=0.05,
                    created_at=_iso(now - timedelta(days=2)))
        _seed_usage(db, row_id="u3", conversation_id="c1", model="claude",
                    tokens_in=400, tokens_out=200, cost_usd=0.40,
                    created_at=_iso(now - timedelta(days=3)))

        client = TestClient(app)
        resp = client.get("/api/usage/summary", headers=_auth())
        assert resp.status_code == 200
        body = resp.json()

        assert body["total"]["input_tokens"]  == 700
        assert body["total"]["output_tokens"] == 330
        assert body["total"]["turns"]         == 3
        assert body["total"]["cost_usd"] == pytest.approx(0.55)


class TestGroupByDay:
    def test_produces_one_row_per_day(self, app, in_memory_db):
        db = in_memory_db
        now = datetime.now(timezone.utc)
        # Three rows on day-1 (same UTC date), two on day-3 (same UTC date).
        day1 = now - timedelta(days=1)
        day3 = now - timedelta(days=3)
        for i in range(3):
            _seed_usage(
                db, row_id=f"u_d1_{i}", conversation_id=None, model="claude",
                tokens_in=10, tokens_out=5, cost_usd=0.01,
                created_at=_iso(day1 + timedelta(seconds=i)),
            )
        for i in range(2):
            _seed_usage(
                db, row_id=f"u_d3_{i}", conversation_id=None, model="claude",
                tokens_in=20, tokens_out=10, cost_usd=0.02,
                created_at=_iso(day3 + timedelta(seconds=i)),
            )

        client = TestClient(app)
        resp = client.get(
            "/api/usage/summary?days=30&group_by=day", headers=_auth(),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["group_by"] == "day"
        # Two distinct UTC dates → two row buckets.
        assert len(body["rows"]) == 2
        keys = [r["key"] for r in body["rows"]]
        # Keys are sorted ascending (oldest first).
        assert keys == sorted(keys)
        # day1 has 3 turns, day3 has 2 turns
        turns_by_key = {r["key"]: r["turns"] for r in body["rows"]}
        assert sorted(turns_by_key.values()) == [2, 3]


class TestGroupByModel:
    def test_groups_correctly(self, app, in_memory_db):
        db = in_memory_db
        now = datetime.now(timezone.utc)
        recent = _iso(now - timedelta(days=2))
        _seed_usage(db, row_id="u1", conversation_id=None, model="claude",
                    tokens_in=100, tokens_out=50, cost_usd=0.10,
                    created_at=recent)
        _seed_usage(db, row_id="u2", conversation_id=None, model="claude",
                    tokens_in=200, tokens_out=80, cost_usd=0.20,
                    created_at=recent)
        _seed_usage(db, row_id="u3", conversation_id=None, model="qwen3",
                    tokens_in=400, tokens_out=200, cost_usd=0.05,
                    created_at=recent)

        client = TestClient(app)
        resp = client.get(
            "/api/usage/summary?days=30&group_by=model", headers=_auth(),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["group_by"] == "model"
        rows = body["rows"]
        assert len(rows) == 2
        # Sorted by cost_usd DESC: claude 0.30 first, then qwen3 0.05.
        assert rows[0]["key"] == "claude"
        assert rows[0]["turns"] == 2
        assert rows[0]["cost_usd"] == pytest.approx(0.30)
        assert rows[1]["key"] == "qwen3"
        assert rows[1]["turns"] == 1
        assert rows[1]["cost_usd"] == pytest.approx(0.05)


class TestGroupByAgent:
    def test_joins_through_conversations(self, app, in_memory_db):
        db = in_memory_db
        now = datetime.now(timezone.utc)
        recent = _iso(now - timedelta(days=2))
        _seed_conversation(db, "c1", "agent-A")
        _seed_conversation(db, "c2", "agent-B")
        _seed_usage(db, row_id="u1", conversation_id="c1", model="claude",
                    tokens_in=100, tokens_out=50, cost_usd=0.10,
                    created_at=recent)
        _seed_usage(db, row_id="u2", conversation_id="c1", model="claude",
                    tokens_in=200, tokens_out=80, cost_usd=0.20,
                    created_at=recent)
        _seed_usage(db, row_id="u3", conversation_id="c2", model="claude",
                    tokens_in=400, tokens_out=200, cost_usd=0.05,
                    created_at=recent)

        client = TestClient(app)
        resp = client.get(
            "/api/usage/summary?days=30&group_by=agent", headers=_auth(),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["group_by"] == "agent"
        rows = body["rows"]
        # Two distinct agents.
        keys = sorted(r["key"] for r in rows)
        assert keys == ["agent-A", "agent-B"]


# ── Time window ────────────────────────────────────────────────────────────

class TestWindowFilter:
    def test_days_7_returns_subset_of_days_30(self, app, in_memory_db):
        db = in_memory_db
        now = datetime.now(timezone.utc)
        recent = _iso(now - timedelta(days=2))
        old    = _iso(now - timedelta(days=20))   # > 7 days, <= 30

        _seed_usage(db, row_id="u_recent", conversation_id=None, model="claude",
                    tokens_in=10, tokens_out=5, cost_usd=0.10,
                    created_at=recent)
        _seed_usage(db, row_id="u_old", conversation_id=None, model="claude",
                    tokens_in=20, tokens_out=10, cost_usd=0.20,
                    created_at=old)

        client = TestClient(app)
        resp_30 = client.get("/api/usage/summary?days=30", headers=_auth())
        resp_7  = client.get("/api/usage/summary?days=7",  headers=_auth())
        assert resp_30.status_code == 200
        assert resp_7.status_code  == 200

        body_30 = resp_30.json()
        body_7  = resp_7.json()

        assert body_30["window_days"] == 30
        assert body_7["window_days"]  == 7
        # 30 day window includes both rows; 7 day window only the recent one.
        assert body_30["total"]["turns"] == 2
        assert body_7["total"]["turns"]  == 1
        assert body_30["total"]["cost_usd"] == pytest.approx(0.30)
        assert body_7["total"]["cost_usd"]  == pytest.approx(0.10)


# ── Auth ───────────────────────────────────────────────────────────────────

class TestAuth:
    def test_rejects_without_bearer_auth(self, app):
        client = TestClient(app)
        resp = client.get("/api/usage/summary")
        assert resp.status_code == 401
