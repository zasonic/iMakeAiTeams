"""
routes/safety.py — read-only aggregation panel over safety subsystems.

GET /api/safety/summary?days=30
  Returns a dashboard payload aggregated from existing tables:
    escalations, pending_writes, canary_baseline, governance_log,
    router_log, session_facts.

  Each section is best-effort: if a table doesn't exist (older install
  pre-migration) or the query fails, that section returns zeros / empty
  lists rather than failing the whole endpoint. The panel turns invisible
  architecture (security engine, governance, escalation channel, memory
  gate, sleeper canary) into visible product surfaces.

Pure read aggregation — never mutates any underlying table.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter

import db as _db

router = APIRouter()
log = logging.getLogger("safety_routes")


def _cutoff_iso(days: int) -> str:
    """Return an ISO-8601 cutoff timestamp `days` ago in UTC."""
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _safe_fetchone(sql: str, params: tuple) -> sqlite3.Row | None:
    """fetchone that swallows missing-table errors and returns None."""
    try:
        return _db.fetchone(sql, params)
    except sqlite3.OperationalError as exc:
        log.debug("safety summary query (one) failed: %s", exc)
        return None
    except Exception as exc:
        log.warning("safety summary query (one) raised: %s", exc)
        return None


def _safe_fetchall(sql: str, params: tuple) -> list[sqlite3.Row]:
    """fetchall that swallows missing-table errors and returns []."""
    try:
        return _db.fetchall(sql, params)
    except sqlite3.OperationalError as exc:
        log.debug("safety summary query (all) failed: %s", exc)
        return []
    except Exception as exc:
        log.warning("safety summary query (all) raised: %s", exc)
        return []


def _escalations_section(cutoff: str) -> dict:
    row = _safe_fetchone(
        """
        SELECT
            COUNT(*) AS triggered,
            SUM(CASE WHEN decision='approved' THEN 1 ELSE 0 END) AS approved,
            SUM(CASE WHEN decision='denied'   THEN 1 ELSE 0 END) AS denied,
            SUM(CASE WHEN decision IS NULL    THEN 1 ELSE 0 END) AS pending
        FROM escalations
        WHERE triggered_at >= ?
        """,
        (cutoff,),
    )
    if row is None:
        return {"triggered": 0, "approved": 0, "denied": 0, "pending": 0}
    return {
        "triggered": int(row["triggered"] or 0),
        "approved":  int(row["approved"]  or 0),
        "denied":    int(row["denied"]    or 0),
        "pending":   int(row["pending"]   or 0),
    }


def _memory_gate_section(cutoff: str) -> dict:
    """Memory gate aggregates two tables.

    pending_writes captures the contested writes that hit the gate. Auto-
    accepted writes never enter pending_writes — they go straight to
    session_facts with source='auto'. To give the panel a complete picture
    we count both: facts_proposed = pending_writes + auto-accepted facts.
    """
    gate_row = _safe_fetchone(
        """
        SELECT
            COUNT(*) AS gated_total,
            SUM(CASE WHEN decision='approved' THEN 1 ELSE 0 END) AS user_approved,
            SUM(CASE WHEN decision='denied'   THEN 1 ELSE 0 END) AS user_denied,
            SUM(CASE WHEN decision IS NULL    THEN 1 ELSE 0 END) AS pending
        FROM pending_writes
        WHERE proposed_at >= ?
        """,
        (cutoff,),
    )

    auto_row = _safe_fetchone(
        """
        SELECT COUNT(*) AS auto_accepted
        FROM session_facts
        WHERE created_at >= ?
          AND (source = 'auto' OR source IS NULL)
        """,
        (cutoff,),
    )

    gated_total   = int(gate_row["gated_total"]   or 0) if gate_row else 0
    user_approved = int(gate_row["user_approved"] or 0) if gate_row else 0
    user_denied   = int(gate_row["user_denied"]   or 0) if gate_row else 0
    pending       = int(gate_row["pending"]       or 0) if gate_row else 0
    auto_accepted = int(auto_row["auto_accepted"] or 0) if auto_row else 0

    return {
        "facts_proposed": gated_total + auto_accepted,
        "auto_accepted":  auto_accepted,
        "user_approved":  user_approved,
        "user_denied":    user_denied,
        "pending":        pending,
    }


def _canary_section(cutoff: str) -> dict:
    """Canary section.

    Drift alerts fire as SSE events (model_canary_alert) and are not
    persisted to a SQL table — there is no log to aggregate. We expose
    the baseline count so the user can see how many models have a
    fingerprint, and leave alerts_fired / last_alert_at at their
    sentinel zero / null until a future migration adds an alerts log.
    """
    row = _safe_fetchone(
        "SELECT COUNT(*) AS baselines FROM canary_baseline",
        (),
    )
    baselines = int(row["baselines"] or 0) if row else 0
    _ = cutoff  # window not applicable to canary_baseline (one row per model)
    return {
        "baselines":      baselines,
        "alerts_fired":   0,
        "last_alert_at":  None,
    }


def _governance_section(cutoff: str) -> dict:
    """governance_log only stores denials (services/governance.py:538).

    tool_calls_total therefore equals the count of *logged* decisions — in
    practice the same as tool_calls_denied — but the query is written
    explicitly so a future change that starts logging successes still
    produces the right ratio.
    """
    counts = _safe_fetchone(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN allowed=0 THEN 1 ELSE 0 END) AS denied
        FROM governance_log
        WHERE created_at >= ?
        """,
        (cutoff,),
    )
    reasons = _safe_fetchall(
        """
        SELECT reason, COUNT(*) AS count
        FROM governance_log
        WHERE allowed = 0
          AND created_at >= ?
          AND reason IS NOT NULL
          AND reason != ''
        GROUP BY reason
        ORDER BY count DESC, reason ASC
        LIMIT 5
        """,
        (cutoff,),
    )
    return {
        "tool_calls_total":  int(counts["total"]  or 0) if counts else 0,
        "tool_calls_denied": int(counts["denied"] or 0) if counts else 0,
        "denial_top_reasons": [
            {"reason": r["reason"], "count": int(r["count"])} for r in reasons
        ],
    }


def _routing_section(cutoff: str) -> dict:
    """Routing section reads router_log.

    A turn is "failed" when either had_error=1 or response_empty=1 — the
    chat_orchestrator records both and the surface should reflect either.
    """
    counts = _safe_fetchone(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN had_error=1 OR response_empty=1 THEN 1 ELSE 0 END) AS failed
        FROM router_log
        WHERE created_at >= ?
        """,
        (cutoff,),
    )
    breakdown = _safe_fetchall(
        """
        SELECT mast_category AS category, COUNT(*) AS count
        FROM router_log
        WHERE created_at >= ?
          AND mast_category IS NOT NULL
          AND mast_category != ''
        GROUP BY mast_category
        ORDER BY count DESC, category ASC
        """,
        (cutoff,),
    )
    return {
        "turns_total":  int(counts["total"]  or 0) if counts else 0,
        "turns_failed": int(counts["failed"] or 0) if counts else 0,
        "mast_breakdown": [
            {"category": r["category"], "count": int(r["count"])} for r in breakdown
        ],
    }


def _voting_section(cutoff: str) -> dict:
    """Voting section reads router_log.voting_samples_json.

    A row carries voting_samples_json only when high-stakes consensus ran.
    Each sample dict has an `all_diverged` flag: when false at least two
    of the three samples agreed → consensus was reached. We use json_extract
    on the first sample (the flag is identical across the three).
    """
    row = _safe_fetchone(
        """
        SELECT
            COUNT(*) AS high_stakes_turns,
            SUM(
                CASE WHEN json_extract(voting_samples_json, '$[0].all_diverged') = 0
                THEN 1 ELSE 0 END
            ) AS consensus_reached
        FROM router_log
        WHERE created_at >= ?
          AND voting_samples_json IS NOT NULL
        """,
        (cutoff,),
    )
    if row is None:
        return {"high_stakes_turns": 0, "consensus_reached": 0}
    return {
        "high_stakes_turns": int(row["high_stakes_turns"] or 0),
        "consensus_reached": int(row["consensus_reached"] or 0),
    }


@router.get("/summary")
async def summary(days: int = 30) -> dict:
    """Aggregate safety counts across subsystems for the last `days` days.

    `days` is clamped to [1, 365] so a malicious or misconfigured caller
    can't request a 100-year window that scans the full history.
    """
    days = max(1, min(int(days or 30), 365))
    cutoff = _cutoff_iso(days)

    return {
        "window_days":  days,
        "escalations":  _escalations_section(cutoff),
        "memory_gate":  _memory_gate_section(cutoff),
        "canary":       _canary_section(cutoff),
        "governance":   _governance_section(cutoff),
        "routing":      _routing_section(cutoff),
        "voting":       _voting_section(cutoff),
    }
