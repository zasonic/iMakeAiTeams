"""
services/turn_lifecycle.py — Turn-level transaction boundary.

The second extraction in the Layer 3 decomposition. Owns the bookend
operations that bracket every chat turn:

  open(ctx)                 — budget check + user-message INSERT,
                              both under the same db lock.
  close(ctx, response, ...) — assistant message + conversation update +
                              token_usage row, all inside one
                              `_db.transaction()` so the three are
                              atomic from the user's point of view.
                              Re-reads `SUM(cost_usd)` inside the same
                              transaction for the budget warning.

The transaction shape here was Bug 5 (race on stale `spent`) and Bug 6
(two commits leaving the DB in a torn state) from the Layer 1 audit.
Moving it into a named module makes the contract explicit and keeps the
orchestrator from re-acquiring `_db._lock` in two different places.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import db as _db
from services.redact import redact
from services.turn_context import TurnContext

log = logging.getLogger("iMakeAiTeams.turn_lifecycle")


@dataclass(frozen=True)
class CloseResult:
    """Outcome of TurnLifecycle.close().

    ``budget_warning`` is the user-visible string when the post-turn spend
    crosses ``warn_pct`` of ``budget``; empty string otherwise. The caller
    threads it into ChatResult.budget_warning.
    """
    budget_warning: str = ""


class TurnLifecycle:
    """Owns the open / close transaction boundary of a chat turn."""

    def __init__(self, settings, local_client=None):
        self._settings = settings
        # Local client is optional — only used by the auto-title side-effect,
        # which is outside the transaction. None disables auto-titling but
        # leaves the rest of the lifecycle intact.
        self._local = local_client

    # ── open ────────────────────────────────────────────────────────────

    def open(self, ctx: TurnContext) -> bool:
        """Budget check + user-message INSERT, atomically.

        Mutates ``ctx`` in place: sets ``user_msg_id``, ``budget``,
        ``warn_pct``, ``spent``, ``budget_exceeded``. Returns ``True``
        iff the turn may proceed (i.e. budget was not exceeded). When
        ``False`` the caller short-circuits to a budget-exceeded
        ChatResult and skips everything else.

        Holds ``_db._lock`` for the SELECT + INSERT so two concurrent
        sends on the same conversation cannot both pass the cap before
        either has recorded its user message.
        """
        ctx.budget = self._settings.get("max_conversation_budget_usd", 5.0)
        ctx.warn_pct = self._settings.get("budget_warning_threshold_pct", 80.0)
        ctx.user_msg_id = str(uuid.uuid4())
        # Layer C1: per-turn correlation id. Generated here so every
        # downstream emit/audit/persist call site reads the same value
        # off the context. Distinct from user_msg_id because a single
        # turn may span multiple messages (e.g. reader/actor split,
        # high-stakes-voting consensus) but only one correlation id.
        ctx.turn_id = str(uuid.uuid4())
        ctx.budget_exceeded = False
        now = datetime.now(timezone.utc).isoformat()

        with _db._lock:
            conn = _db.get_db()
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) as total FROM token_usage "
                "WHERE conversation_id = ?",
                (ctx.conversation_id,),
            ).fetchone()
            ctx.spent = row["total"] if row else 0.0
            if ctx.budget > 0 and ctx.spent >= ctx.budget:
                ctx.budget_exceeded = True
                return False
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content, created_at) "
                "VALUES (?, ?, 'user', ?, ?)",
                (ctx.user_msg_id, ctx.conversation_id, ctx.user_message, now),
            )
            conn.commit()
        return True

    # ── close ───────────────────────────────────────────────────────────

    def close(
        self,
        ctx: TurnContext,
        asst_msg_id: str,
        response_text: str,
        route_reason: str,
        model_name: str,
        tokens_in: int,
        tokens_out: int,
        cost: float,
    ) -> CloseResult:
        """Persist the assistant turn atomically.

        Three writes go into ONE _db.transaction() so a crash between
        them cannot leave the conversation with a message but no matching
        token_usage row (the original Bug 6 failure mode):

          1. INSERT INTO messages (assistant)
          2. UPDATE conversations SET updated_at, title = …
          3. INSERT INTO token_usage

        The budget warning re-reads SUM(cost_usd) inside the same
        transaction (closing Bug 5's race on the pre-turn ``ctx.spent``
        snapshot).

        Auto-title is INTENTIONALLY outside the transaction: it makes a
        blocking LLM call we don't want to hold locks across. The
        helper ``maybe_auto_title`` runs after commit.
        """
        reply_text_for_storage = redact(response_text)
        resp_now = datetime.now(timezone.utc).isoformat()
        budget_warning = ""

        with _db.transaction() as conn:
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content, model_used, "
                "route_reason, tokens_in, tokens_out, cost_usd, created_at) "
                "VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?, ?, ?)",
                (
                    asst_msg_id, ctx.conversation_id, reply_text_for_storage,
                    model_name, route_reason, tokens_in, tokens_out, cost,
                    resp_now,
                ),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ?, "
                "title = CASE WHEN title = 'New conversation' THEN ? ELSE title END "
                "WHERE id = ?",
                (resp_now, ctx.user_message[:60], ctx.conversation_id),
            )
            conn.execute(
                "INSERT INTO token_usage (id, conversation_id, model, tokens_in, "
                "tokens_out, cost_usd, routed_reason, created_at, turn_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()), ctx.conversation_id, model_name,
                    tokens_in, tokens_out, cost, route_reason, resp_now,
                    ctx.turn_id,
                ),
            )
            if ctx.budget > 0:
                row = conn.execute(
                    "SELECT COALESCE(SUM(cost_usd), 0) as total FROM token_usage "
                    "WHERE conversation_id = ?",
                    (ctx.conversation_id,),
                ).fetchone()
                # Fall back to the pre-turn snapshot + this turn's cost only
                # when the SUM somehow returns no row — which shouldn't
                # happen now that we just INSERTed, but the fallback is
                # cheap and matches the pre-Layer-3 behavior.
                new_spent = row["total"] if row else (ctx.spent + cost)
                pct = (new_spent / ctx.budget) * 100
                if pct >= ctx.warn_pct:
                    budget_warning = (
                        f"⚠️ Approaching conversation budget limit "
                        f"(${new_spent:.2f}/${ctx.budget:.2f})"
                    )

        return CloseResult(budget_warning=budget_warning)

    # ── auto-title (post-commit, best-effort) ───────────────────────────

    def maybe_auto_title(self, ctx: TurnContext, response_text: str) -> None:
        """Generate a short title from the first exchange, in place.

        Fires only when the conversation's title is still the raw user
        message (i.e. this is the first assistant reply). Uses the local
        client; failures are swallowed because titling is cosmetic.
        """
        if self._local is None:
            return
        try:
            if not self._local.is_available():
                return
        except Exception:
            return

        conv_row = _db.fetchone(
            "SELECT title FROM conversations WHERE id = ?",
            (ctx.conversation_id,),
        )
        if not conv_row or conv_row["title"] != ctx.user_message[:60]:
            return

        try:
            title_raw = self._local.chat(
                "Generate a 3-6 word title for this conversation. "
                "Return ONLY the title text, no quotes, no explanation.",
                f"User: {ctx.user_message[:200]}\nAssistant: {response_text[:200]}",
                max_tokens=20,
            )
        except Exception as exc:
            log.debug("auto-title local.chat failed: %s", exc)
            return

        if not title_raw:
            return
        clean = title_raw.strip().strip('"\'').strip()
        if not (2 < len(clean) <= 80):
            return
        try:
            _db.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (clean, ctx.conversation_id),
            )
            _db.commit()
        except Exception as exc:
            log.debug("auto-title UPDATE failed: %s", exc)
