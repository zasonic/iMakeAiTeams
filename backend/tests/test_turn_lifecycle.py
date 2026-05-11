"""
tests/test_turn_lifecycle.py — Layer 3: TurnLifecycle module.

Pins the transaction shape Layer 1 hardened so the extraction can't
quietly regress Bug 5 (race on stale ``spent``) or Bug 6 (two commits
leaving the DB in a torn state). Also covers auto-title's local-client
gating — a path that's easy to break in a refactor.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from services.turn_context import TurnContext
from services.turn_lifecycle import CloseResult, TurnLifecycle


def _seed_conversation(in_memory_db, title: str = "Hello there") -> str:
    cid = str(uuid.uuid4())
    in_memory_db.execute(
        "INSERT INTO conversations (id, title, agent_id, created_at, updated_at) "
        "VALUES (?, ?, NULL, '2024-01-01', '2024-01-01')",
        (cid, title),
    )
    in_memory_db.commit()
    return cid


@pytest.fixture
def lifecycle(settings):
    return TurnLifecycle(settings)


# ── open ─────────────────────────────────────────────────────────────────────


def test_open_inserts_user_message_and_returns_true(in_memory_db, lifecycle):
    cid = _seed_conversation(in_memory_db)
    ctx = TurnContext(conversation_id=cid, user_message="hello")
    assert lifecycle.open(ctx) is True
    assert ctx.user_msg_id != ""
    assert ctx.budget_exceeded is False
    rows = in_memory_db.fetchall(
        "SELECT id, role, content FROM messages WHERE conversation_id = ?",
        (cid,),
    )
    assert len(rows) == 1
    assert rows[0]["role"] == "user"
    assert rows[0]["content"] == "hello"
    assert rows[0]["id"] == ctx.user_msg_id


def test_open_assigns_turn_id(in_memory_db, lifecycle):
    """Layer C1: every open() sets ctx.turn_id to a fresh UUID. Distinct
    from user_msg_id (a single turn can produce multiple message rows
    via reader/actor/voting but only one correlation id)."""
    cid = _seed_conversation(in_memory_db)
    ctx = TurnContext(conversation_id=cid, user_message="hello")
    assert ctx.turn_id == ""  # populated by open(), not at construction
    lifecycle.open(ctx)
    assert ctx.turn_id != ""
    assert ctx.turn_id != ctx.user_msg_id  # two separate ids
    # Format check: stringified UUID
    assert len(ctx.turn_id) == 36 and ctx.turn_id.count("-") == 4


def test_open_assigns_unique_turn_id_per_call(in_memory_db, lifecycle):
    """Two opens on the same conversation get different turn_ids."""
    cid = _seed_conversation(in_memory_db)
    ctx1 = TurnContext(conversation_id=cid, user_message="first")
    lifecycle.open(ctx1)
    ctx2 = TurnContext(conversation_id=cid, user_message="second")
    lifecycle.open(ctx2)
    assert ctx1.turn_id != ctx2.turn_id


def test_open_blocks_when_spent_meets_budget(in_memory_db, lifecycle, settings):
    cid = _seed_conversation(in_memory_db)
    settings.set("max_conversation_budget_usd", 1.0)
    # Pre-seed token_usage so ``spent`` already equals the budget.
    in_memory_db.execute(
        "INSERT INTO token_usage (id, conversation_id, model, tokens_in, "
        "tokens_out, cost_usd, routed_reason, created_at) "
        "VALUES (?, ?, 'm', 0, 0, 1.0, 'r', '2024-01-01')",
        (str(uuid.uuid4()), cid),
    )
    in_memory_db.commit()

    ctx = TurnContext(conversation_id=cid, user_message="please")
    assert lifecycle.open(ctx) is False
    assert ctx.budget_exceeded is True
    # No user message should have been inserted on the budget-block path.
    rows = in_memory_db.fetchall(
        "SELECT id FROM messages WHERE conversation_id = ?", (cid,),
    )
    assert rows == []


def test_open_records_pre_turn_snapshot(in_memory_db, lifecycle, settings):
    cid = _seed_conversation(in_memory_db)
    settings.set("max_conversation_budget_usd", 5.0)
    in_memory_db.execute(
        "INSERT INTO token_usage (id, conversation_id, model, tokens_in, "
        "tokens_out, cost_usd, routed_reason, created_at) "
        "VALUES (?, ?, 'm', 0, 0, 1.25, 'r', '2024-01-01')",
        (str(uuid.uuid4()), cid),
    )
    in_memory_db.commit()

    ctx = TurnContext(conversation_id=cid, user_message="yo")
    assert lifecycle.open(ctx) is True
    assert ctx.spent == pytest.approx(1.25)
    assert ctx.budget == pytest.approx(5.0)


# ── close ────────────────────────────────────────────────────────────────────


def test_close_writes_three_rows_atomically(in_memory_db, lifecycle):
    """Bug 6 regression: assistant message + token_usage + conversation
    update commit together so a crash between writes can't tear the DB."""
    cid = _seed_conversation(in_memory_db)
    ctx = TurnContext(conversation_id=cid, user_message="hi")
    lifecycle.open(ctx)
    asst_msg_id = str(uuid.uuid4())

    result = lifecycle.close(
        ctx,
        asst_msg_id=asst_msg_id,
        response_text="reply",
        route_reason="test",
        model_name="m",
        tokens_in=10,
        tokens_out=20,
        cost=0.05,
    )

    assert isinstance(result, CloseResult)
    msgs = in_memory_db.fetchall(
        "SELECT id, role, content, model_used FROM messages WHERE conversation_id = ?",
        (cid,),
    )
    # User msg from open() + assistant msg from close()
    assert len(msgs) == 2
    asst_rows = [m for m in msgs if m["role"] == "assistant"]
    assert len(asst_rows) == 1
    assert asst_rows[0]["id"] == asst_msg_id
    assert asst_rows[0]["content"] == "reply"

    usage = in_memory_db.fetchall(
        "SELECT cost_usd, model, turn_id FROM token_usage WHERE conversation_id = ?",
        (cid,),
    )
    assert len(usage) == 1
    assert usage[0]["cost_usd"] == pytest.approx(0.05)
    # Layer C1: token_usage row carries the per-turn correlation id.
    assert usage[0]["turn_id"] == ctx.turn_id


def test_close_rolls_back_on_mid_transaction_error(in_memory_db, lifecycle, monkeypatch):
    """If the token_usage INSERT raises, the assistant message INSERT and
    conversation UPDATE must roll back — leaving zero new rows, not one."""
    cid = _seed_conversation(in_memory_db)
    ctx = TurnContext(conversation_id=cid, user_message="hi")
    lifecycle.open(ctx)
    pre_count = in_memory_db.fetchone(
        "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ? AND role = 'assistant'",
        (cid,),
    )["n"]

    # Force the third INSERT (token_usage) to fail mid-transaction. We
    # can't reassign sqlite3.Connection.execute (read-only C attribute),
    # so wrap the connection in a proxy that delegates everything except
    # ``execute``.
    import db
    real_transaction = db.transaction
    raised = {"hit": False}

    class _ProxyConn:
        def __init__(self, real):
            self._real = real
        def execute(self, sql, params=()):
            if "INSERT INTO token_usage" in sql and not raised["hit"]:
                raised["hit"] = True
                raise RuntimeError("simulated mid-write failure")
            return self._real.execute(sql, params)
        def commit(self):
            return self._real.commit()
        def rollback(self):
            return self._real.rollback()
        def __getattr__(self, name):
            return getattr(self._real, name)

    from contextlib import contextmanager
    @contextmanager
    def _flaky():
        with real_transaction() as conn:
            yield _ProxyConn(conn)
    monkeypatch.setattr(db, "transaction", _flaky)

    with pytest.raises(RuntimeError):
        lifecycle.close(
            ctx, asst_msg_id=str(uuid.uuid4()), response_text="r",
            route_reason="t", model_name="m",
            tokens_in=1, tokens_out=1, cost=0.01,
        )

    post_count = in_memory_db.fetchone(
        "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ? AND role = 'assistant'",
        (cid,),
    )["n"]
    assert post_count == pre_count, "assistant message must roll back when token_usage fails"
    usage = in_memory_db.fetchall(
        "SELECT id FROM token_usage WHERE conversation_id = ?", (cid,),
    )
    assert usage == []


def test_close_budget_warning_uses_post_turn_total(in_memory_db, lifecycle, settings):
    """Bug 5 regression: budget warning re-reads SUM inside the txn so
    concurrent sends can't race on a stale pre-turn snapshot."""
    cid = _seed_conversation(in_memory_db)
    settings.set("max_conversation_budget_usd", 1.0)
    settings.set("budget_warning_threshold_pct", 80.0)
    # Pre-seed 0.50 of spend — pre-turn ``spent`` will be 0.50.
    in_memory_db.execute(
        "INSERT INTO token_usage (id, conversation_id, model, tokens_in, "
        "tokens_out, cost_usd, routed_reason, created_at) "
        "VALUES (?, ?, 'm', 0, 0, 0.50, 'r', '2024-01-01')",
        (str(uuid.uuid4()), cid),
    )
    in_memory_db.commit()

    ctx = TurnContext(conversation_id=cid, user_message="hi")
    lifecycle.open(ctx)
    # This turn adds 0.40, putting cumulative at 0.90 — past 80% of $1.00.
    result = lifecycle.close(
        ctx, asst_msg_id=str(uuid.uuid4()),
        response_text="r", route_reason="t", model_name="m",
        tokens_in=0, tokens_out=0, cost=0.40,
    )
    assert "$0.90" in result.budget_warning
    assert "$1.00" in result.budget_warning


def test_close_no_warning_below_threshold(in_memory_db, lifecycle, settings):
    cid = _seed_conversation(in_memory_db)
    settings.set("max_conversation_budget_usd", 5.0)
    settings.set("budget_warning_threshold_pct", 80.0)
    ctx = TurnContext(conversation_id=cid, user_message="hi")
    lifecycle.open(ctx)
    result = lifecycle.close(
        ctx, asst_msg_id=str(uuid.uuid4()),
        response_text="r", route_reason="t", model_name="m",
        tokens_in=0, tokens_out=0, cost=0.01,
    )
    assert result.budget_warning == ""


# ── auto-title ───────────────────────────────────────────────────────────────


def test_auto_title_replaces_truncated_title(in_memory_db, settings):
    cid = _seed_conversation(in_memory_db, title="hello there")
    local = MagicMock()
    local.is_available.return_value = True
    local.chat.return_value = "  Quick Greeting  "
    lc = TurnLifecycle(settings, local_client=local)
    ctx = TurnContext(conversation_id=cid, user_message="hello there")
    lc.maybe_auto_title(ctx, "Hi! How can I help?")
    row = in_memory_db.fetchone(
        "SELECT title FROM conversations WHERE id = ?", (cid,),
    )
    assert row["title"] == "Quick Greeting"


def test_auto_title_skipped_when_local_unavailable(in_memory_db, settings):
    cid = _seed_conversation(in_memory_db, title="hello there")
    local = MagicMock()
    local.is_available.return_value = False
    lc = TurnLifecycle(settings, local_client=local)
    ctx = TurnContext(conversation_id=cid, user_message="hello there")
    lc.maybe_auto_title(ctx, "reply")
    local.chat.assert_not_called()
    row = in_memory_db.fetchone(
        "SELECT title FROM conversations WHERE id = ?", (cid,),
    )
    assert row["title"] == "hello there"


def test_auto_title_skipped_when_title_already_changed(in_memory_db, settings):
    cid = _seed_conversation(in_memory_db, title="Manually Set Title")
    local = MagicMock()
    local.is_available.return_value = True
    lc = TurnLifecycle(settings, local_client=local)
    ctx = TurnContext(conversation_id=cid, user_message="this is a different msg")
    lc.maybe_auto_title(ctx, "reply")
    local.chat.assert_not_called()


def test_auto_title_skipped_when_no_local_client(in_memory_db, settings):
    cid = _seed_conversation(in_memory_db, title="hello there")
    lc = TurnLifecycle(settings)  # no local
    ctx = TurnContext(conversation_id=cid, user_message="hello there")
    # Must not raise.
    lc.maybe_auto_title(ctx, "reply")


def test_auto_title_swallows_local_exception(in_memory_db, settings):
    cid = _seed_conversation(in_memory_db, title="hello there")
    local = MagicMock()
    local.is_available.return_value = True
    local.chat.side_effect = RuntimeError("boom")
    lc = TurnLifecycle(settings, local_client=local)
    ctx = TurnContext(conversation_id=cid, user_message="hello there")
    # Must not raise — auto-title is best-effort.
    lc.maybe_auto_title(ctx, "reply")
