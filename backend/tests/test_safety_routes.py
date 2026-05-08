"""tests/test_safety_routes.py — HTTP-level tests for /api/safety routes.

Exercises the read-only aggregation endpoint via FastAPI's TestClient.
Builds a minimal app mounted on top of the in_memory_db fixture so the
queries hit a real (temp) SQLite file and we exercise the same code
path the renderer talks to.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routes import safety as safety_routes
from server import BearerAuthMiddleware


TOKEN = "test-token-safety"


def _auth() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


@pytest.fixture
def app(in_memory_db):
    a = FastAPI()
    a.add_middleware(BearerAuthMiddleware, expected_token=TOKEN)
    a.include_router(safety_routes.router, prefix="/api/safety")
    return a


# ── Empty database ─────────────────────────────────────────────────────────

class TestEmptyDatabase:
    def test_returns_zeros_and_empty_lists(self, app):
        client = TestClient(app)
        resp = client.get("/api/safety/summary", headers=_auth())
        assert resp.status_code == 200
        body = resp.json()

        assert body["window_days"] == 30
        assert body["escalations"] == {
            "triggered": 0, "approved": 0, "denied": 0, "pending": 0,
        }
        assert body["memory_gate"] == {
            "facts_proposed": 0, "auto_accepted": 0,
            "user_approved":  0, "user_denied":   0, "pending": 0,
        }
        assert body["canary"]["baselines"] == 0
        assert body["canary"]["alerts_fired"] == 0
        assert body["canary"]["last_alert_at"] is None
        assert body["governance"]["tool_calls_total"]   == 0
        assert body["governance"]["tool_calls_denied"]  == 0
        assert body["governance"]["denial_top_reasons"] == []
        assert body["routing"]["turns_total"]    == 0
        assert body["routing"]["turns_failed"]   == 0
        assert body["routing"]["mast_breakdown"] == []
        assert body["voting"]["high_stakes_turns"] == 0
        assert body["voting"]["consensus_reached"] == 0


# ── Seeded database ────────────────────────────────────────────────────────

def _seed_within_window(in_memory_db) -> None:
    """Seed every relevant table with rows inside a 30-day window."""
    db = in_memory_db
    now = datetime.now(timezone.utc)
    recent = _iso(now - timedelta(days=2))

    # escalations: 1 approved, 1 denied, 1 pending
    db.execute(
        "INSERT INTO escalations (id, conversation_id, triggered_at, "
        "trigger_type, decision) VALUES (?, ?, ?, ?, ?)",
        ("e1", "c1", recent, "replacement_threat", "approved"),
    )
    db.execute(
        "INSERT INTO escalations (id, conversation_id, triggered_at, "
        "trigger_type, decision) VALUES (?, ?, ?, ?, ?)",
        ("e2", "c1", recent, "autonomy_reduction", "denied"),
    )
    db.execute(
        "INSERT INTO escalations (id, conversation_id, triggered_at, "
        "trigger_type, decision) VALUES (?, ?, ?, ?, ?)",
        ("e3", "c2", recent, "goal_conflict", None),
    )
    db.commit()

    # pending_writes: 1 approved, 1 denied, 1 pending
    db.execute(
        "INSERT INTO pending_writes (id, conversation_id, write_type, "
        "content, proposed_at, decision) VALUES (?, ?, ?, ?, ?, ?)",
        ("pw1", "c1", "fact", "f1", recent, "approved"),
    )
    db.execute(
        "INSERT INTO pending_writes (id, conversation_id, write_type, "
        "content, proposed_at, decision) VALUES (?, ?, ?, ?, ?, ?)",
        ("pw2", "c1", "fact", "f2", recent, "denied"),
    )
    db.execute(
        "INSERT INTO pending_writes (id, conversation_id, write_type, "
        "content, proposed_at, decision) VALUES (?, ?, ?, ?, ?, ?)",
        ("pw3", "c1", "fact", "f3", recent, None),
    )
    db.commit()

    # session_facts: 2 auto-accepted within window. conversation_id is left
    # NULL so we don't need to seed the conversations parent row (the column
    # is FK-referenced but NULL is allowed by SQLite for nullable FKs).
    db.execute(
        "INSERT INTO session_facts (id, conversation_id, fact, source, "
        "created_at) VALUES (?, ?, ?, ?, ?)",
        ("sf1", None, "auto-fact-1", "auto", recent),
    )
    db.execute(
        "INSERT INTO session_facts (id, conversation_id, fact, source, "
        "created_at) VALUES (?, ?, ?, ?, ?)",
        ("sf2", None, "auto-fact-2", "auto", recent),
    )
    db.commit()

    # canary_baseline: 2 baselines (no created_at filter applies here)
    db.execute(
        "INSERT INTO canary_baseline (id, model_id, prompt_hash, "
        "prompt_text, response_text, embedding, captured_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("cb1", "qwen3", "h1", "p1", "r1", b"\x00\x00", recent),
    )
    db.execute(
        "INSERT INTO canary_baseline (id, model_id, prompt_hash, "
        "prompt_text, response_text, embedding, captured_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("cb2", "qwen3", "h2", "p2", "r2", b"\x00\x00", recent),
    )
    db.commit()

    # governance_log: 3 denials with 2 reasons + 1 allowed
    db.execute(
        "INSERT INTO governance_log (id, agent_id, tool_name, allowed, "
        "reason, policy_name, task_key, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("g1", "a1", "shell", 0, "tool not in allowlist", "p1", "t1", recent),
    )
    db.execute(
        "INSERT INTO governance_log (id, agent_id, tool_name, allowed, "
        "reason, policy_name, task_key, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("g2", "a1", "shell", 0, "tool not in allowlist", "p1", "t2", recent),
    )
    db.execute(
        "INSERT INTO governance_log (id, agent_id, tool_name, allowed, "
        "reason, policy_name, task_key, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("g3", "a2", "fs.write", 0, "rate limit exceeded", "p2", "t3", recent),
    )
    db.execute(
        "INSERT INTO governance_log (id, agent_id, tool_name, allowed, "
        "reason, policy_name, task_key, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("g4", "a2", "fs.read", 1, "ok", "p2", "t4", recent),
    )
    db.commit()

    # router_log: 4 turns, 1 failed (had_error), 1 with mast_category, 1 voting
    db.execute(
        "INSERT INTO router_log (id, conversation_id, message_preview, "
        "route_taken, complexity, reasoning, tokens_out, had_error, "
        "response_empty, model_used, created_at, mast_category, agent_role, "
        "voting_samples_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("r1", "c1", "hello", "claude", "low", "ok",
         100, 0, 0, "claude-sonnet", recent, None, "monolithic", None),
    )
    db.execute(
        "INSERT INTO router_log (id, conversation_id, message_preview, "
        "route_taken, complexity, reasoning, tokens_out, had_error, "
        "response_empty, model_used, created_at, mast_category, agent_role, "
        "voting_samples_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("r2", "c1", "broken", "local", "low", "ok",
         0, 1, 0, "qwen3", recent, "tool_misuse", "monolithic", None),
    )
    db.execute(
        "INSERT INTO router_log (id, conversation_id, message_preview, "
        "route_taken, complexity, reasoning, tokens_out, had_error, "
        "response_empty, model_used, created_at, mast_category, agent_role, "
        "voting_samples_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("r3", "c2", "vote-ok", "claude", "high", "stakes",
         200, 0, 0, "claude-opus", recent, None, "monolithic",
         '[{"all_diverged": false, "chosen": true},'
         ' {"all_diverged": false, "chosen": false},'
         ' {"all_diverged": false, "chosen": false}]'),
    )
    db.execute(
        "INSERT INTO router_log (id, conversation_id, message_preview, "
        "route_taken, complexity, reasoning, tokens_out, had_error, "
        "response_empty, model_used, created_at, mast_category, agent_role, "
        "voting_samples_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("r4", "c2", "vote-split", "claude", "high", "stakes",
         200, 0, 0, "claude-opus", recent, None, "monolithic",
         '[{"all_diverged": true, "chosen": true},'
         ' {"all_diverged": true, "chosen": false},'
         ' {"all_diverged": true, "chosen": false}]'),
    )
    db.commit()


class TestSeededAggregation:
    def test_aggregates_each_section_correctly(self, app, in_memory_db):
        _seed_within_window(in_memory_db)
        client = TestClient(app)
        resp = client.get("/api/safety/summary", headers=_auth())
        assert resp.status_code == 200
        body = resp.json()

        assert body["escalations"] == {
            "triggered": 3, "approved": 1, "denied": 1, "pending": 1,
        }
        assert body["memory_gate"] == {
            "facts_proposed": 5,    # 3 pending_writes + 2 auto session_facts
            "auto_accepted":  2,
            "user_approved":  1,
            "user_denied":    1,
            "pending":        1,
        }
        assert body["canary"]["baselines"] == 2
        # governance_log: 4 logged, 3 denied, top reason is "tool not in allowlist" x2
        assert body["governance"]["tool_calls_total"]  == 4
        assert body["governance"]["tool_calls_denied"] == 3
        reasons = body["governance"]["denial_top_reasons"]
        assert reasons[0] == {"reason": "tool not in allowlist", "count": 2}
        assert {"reason": "rate limit exceeded", "count": 1} in reasons
        # router_log: 4 turns, 1 failed (had_error), 1 with mast_category
        assert body["routing"]["turns_total"]  == 4
        assert body["routing"]["turns_failed"] == 1
        assert body["routing"]["mast_breakdown"] == [
            {"category": "tool_misuse", "count": 1}
        ]
        # voting: 2 high-stakes turns, 1 with consensus (all_diverged=false)
        assert body["voting"]["high_stakes_turns"] == 2
        assert body["voting"]["consensus_reached"] == 1


# ── Time window ────────────────────────────────────────────────────────────

class TestWindowFilter:
    def test_days_7_excludes_older_rows_that_days_30_includes(self, app, in_memory_db):
        db = in_memory_db
        now = datetime.now(timezone.utc)
        recent = _iso(now - timedelta(days=2))
        old    = _iso(now - timedelta(days=20))  # > 7 days, <= 30 days

        db.execute(
            "INSERT INTO escalations (id, conversation_id, triggered_at, "
            "trigger_type, decision) VALUES (?, ?, ?, ?, ?)",
            ("e_recent", "c", recent, "replacement_threat", "approved"),
        )
        db.execute(
            "INSERT INTO escalations (id, conversation_id, triggered_at, "
            "trigger_type, decision) VALUES (?, ?, ?, ?, ?)",
            ("e_old", "c", old, "replacement_threat", "approved"),
        )
        db.commit()

        client = TestClient(app)

        resp_30 = client.get("/api/safety/summary?days=30", headers=_auth())
        resp_7  = client.get("/api/safety/summary?days=7",  headers=_auth())
        assert resp_30.status_code == 200
        assert resp_7.status_code  == 200

        body_30 = resp_30.json()
        body_7  = resp_7.json()

        assert body_30["window_days"] == 30
        assert body_7["window_days"]  == 7
        assert body_30["escalations"]["triggered"] == 2
        assert body_7["escalations"]["triggered"]  == 1


# ── Auth ───────────────────────────────────────────────────────────────────

class TestAuth:
    def test_rejects_without_bearer_auth(self, app):
        client = TestClient(app)
        resp = client.get("/api/safety/summary")
        assert resp.status_code == 401
